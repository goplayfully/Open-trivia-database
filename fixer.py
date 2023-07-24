import json
import sys
import logging
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
        2) Fix unnecessary title casing
        3) Remove unncessary spaces near quotation marks
        4) Add three plausible, unique, wrong entries to the `answers` array
        5) Please format the output to a single line without indentation or newlines."""

    filename = sys.argv[1]
    out_file = open("augmented.out", "a")
    input_lines = ""
    BATCH_SIZE = 10
    lines = 0

    with open(filename) as file:
        # skip first line
        next_n_lines = list(islice(file, 1))
        while True:
            logging.info("Processing a batch starting at %s...", lines)
            lines = lines + BATCH_SIZE
            next_n_lines = list(islice(file, BATCH_SIZE))
            if not next_n_lines:
                break
            next_n_lines
            for line in next_n_lines:
              input_lines = input_lines + line

            response = chat.send_message(
              prompt_template + input_lines,
              **parameters)
            logging.info(f"Response from Model: {response.text}")
            out_file.write(response.text + "\n")
            # if lines > 20:
            #     break
    out_file.close()

    # Then please add plausible but wrong answers to the `answers` array, such that answer[0] is the correct one. Finally, p
    # for line in lines:
    #     json_object = parse_json_line(line)
    #     if json_object is None:
    #         print(line)


if __name__ == "__main__":
    main()
