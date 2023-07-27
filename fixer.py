import json
import sys
import random
import logging
import coloredlogs
import hashlib
import vertexai
from itertools import islice
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Firebase for caching
# Use the application default credentials.
cred = credentials.ApplicationDefault()
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
    database_handle = firestore.client()


VERTEX_STORY_MODEL = "chat-bison@001"
# OPENAI_MODEL="gpt-4-0613"
OPENAI_TEMPERATURE = 0.2

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

def read_lines(filename):
    """Reads the contents of a file and returns a list of lines.
    Args:
      filename: The name of the file to read.
    Returns:
      A list of lines in the file.
    """

    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def parse_json_line(line):
    """Parses a line of text into JSON and returns the result.
    Args:
      line: The line of text to parse.
    Returns:
      A JSON object or None if the line could not be parsed.
    """

    # chomp last comma
    line = line[:-2]
    try:
        json_object = json.loads(line)
        return json_object
    except json.JSONDecodeError:
        return None


def main():
    '''Process input files from Open-trivia-database and ask LLM to
    fix formatting problems and add multiple-choice answers'''

    logging.basicConfig(level=logging.DEBUG)

    # Set up Vertex/PaLM
    chat_model = ChatModel.from_pretrained(VERTEX_STORY_MODEL)
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    chat = chat_model.start_chat(
        context="You are a code editor for a set of trivia questions stored as JSON objects. Please fix the following trivia questions.",
        examples=[
            InputOutputTextPair(
                input_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":[], "question":"The phrase "Homo sapiens " means ____", "answer":0, "answers":["Man of knowledge"], "source":""},""",
                output_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":[], "question":"The phrase \"Homo sapiens\" means ____", "answer":0, "answers":["Man of knowledge", "Man of science", "Man who thinks", "Man of steel"], "source":""},""",
            ),
            InputOutputTextPair(
                input_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":["SCIENCE_AND_NATURE"], "question":"__________ and short_tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants"], "source":""}""",
                output_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":["SCIENCE_AND_NATURE"], "question":"__________ and short_tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants", "Horses", "Hippos", "Voles"], "source":""}""",
            )
        ],
    )

    prompt_template = """For each line in the following JSON array, please perform
        the following steps:
        1) Inside the "question" element, escape all
        unescaped or duplicated quotation marks with a \ character so the line is valid JSON
        2) Fix unnecessary title casing and unnecessary _ characters
        3) Remove unncessary spaces near quotation marks
        4) Add three plausible, unique, wrong entries to the `answers` array
        5) Ensure that the "question" is a valid question (and doesn't give away
        the answer)
        6) Please format the output to a single line without indentation or newlines."""

    filename = sys.argv[1]
    out_file = open("augmented.json", "a")
    problem_file = open("problems.json", "a")
    BATCH_SIZE = 10
    lines = 0

    doc_ref = database_handle.collection("trivia")

    with open(filename) as file:
        # skip first line
        next_n_lines = list(islice(file, 1))
        while True:
            logger.info("Processing a batch starting at %s...", lines)
            lines = lines + BATCH_SIZE
            next_n_lines = list(islice(file, BATCH_SIZE))
            if not next_n_lines:
                break

            # create a concatenated string
            input_lines = ""
            for line in next_n_lines:
                input_lines = input_lines + line

            logger.info('Submitting to model: input %s', input_lines)

            response = chat.send_message(
                prompt_template + input_lines,
                **parameters)
            logger.info(f"Response from Model: {response.text}")
            out_file.write(response.text + "\n")

            # write to DB
            for response_line in response.text.splitlines():
                trimmed_line = response_line
                try:
                    # chomp the trailing , so we can parse this individually
                    if trimmed_line[-1] == ',':
                        trimmed_line = trimmed_line[:-1]
                    question_obj = json.loads(trimmed_line)
                except json.JSONDecodeError as err:
                    logger.error(
                        "Could not parse line '%s', %s @ %s",
                        response_line,
                        err.msg,
                        err.pos)
                    problem_file.write(response_line + "\n")

                question_obj["random_1"] = int(random.getrandbits(32))
                question_obj["random_2"] = int(random.getrandbits(32))
                question_obj["random_3"] = int(random.getrandbits(32))
                try:
                    question_obj["correct_answer"] = question_obj["answers"][0]
                except KeyError as err:
                    logger.error("Missing answer in question %s (%s)",
                                 trimmed_line, err)
                    problem_file.write(response_line + "\n")

                # TODO(mrisher): This hashes based on the post-LLM topic
                # so potentially we already have an essentially identical
                # question in the db. In the future, maybe we should hash
                # based on the raw input question (from the json file)
                # but then we would need to map the input to output to
                # get the key
                question_key = hashlib.sha1(
                    question_obj["question"].encode("utf-8")).hexdigest()
                if doc_ref.document(question_key).get().exists:
                    logger.info(
                        'Already found entry with key %s',
                        question_key)
                    continue
                logger.info(
                    "Adding line to database: '%s'",
                    response_line[:-1])
                question_obj["content_id"] = question_key

                # remove "answer" key, which was a complicated array index
                try:
                    question_obj.pop("answer")
                except (KeyError):
                    # ignore
                    logger.debug('question had no "answer" field')
                doc_ref.document(question_key).set(question_obj)

            # early exit
            if lines > 1000:
                break
    out_file.close()
    problem_file.close()

    # Then please add plausible but wrong answers to the `answers` array, such that answer[0] is the correct one. Finally, p
    # for line in lines:
    #     json_object = parse_json_line(line)
    #     if json_object is None:
    #         print(line)


if __name__ == "__main__":
    main()
