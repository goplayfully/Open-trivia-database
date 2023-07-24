import json
import sys
import vertexai
from vertexai.preview.language_models import CodeChatModel


VERTEX_STORY_MODEL = "codechat-bison@001"
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
    filename = sys.argv[1]
    lines = read_lines(filename)

    chat_model = CodeChatModel.from_pretrained(VERTEX_STORY_MODEL)
    parameters = {
      "temperature": 0.2,
      "max_output_tokens": 1024,
    }
    chat = chat_model.start_chat()
    response = chat.send_message(
        """For each line in the following JSON array, please escape the
        problematic unescaped quotation marks with a \ character,
        ensuring the line is now valid JSON. Please format the output
        without indentation or newlines.

    [
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"Homo sapiens\" means:\\\"\", \"answer\":0, \"answers\":[\"Man of knowledge\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How does Alice kill Freddy Krueger in \"Nightmare on Elm Street 4\"?\", \"answer\":0, \"answers\":[\"With a mirror\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is \" Alberto De Salvo \" Better Known?\", \"answer\":0, \"answers\":[\"The Boston Strangler\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is \"Ethylene Glycol\" Better Known?\", \"answer\":0, \"answers\":[\"Antifreeze\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is Singer, Actor, Writer & Producer \" Tracy Marrow \" Better Known?\", \"answer\":0, \"answers\":[\"Ice T\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is The Character Of \" Paul Metcalfe \" Better Known?\", \"answer\":0, \"answers\":[\"Captain Scarlett\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is The Fictional Character The \" Duchess of St Bridget \" Otherwise Known ?\", \"answer\":0, \"answers\":[\"Lara Croft\"], \"source\":\"\"},
      {\"category_id\":\"UNCATEGORIZED\", \"lang\":\"en\", \"tags\":[], \"question\":\"How Is The Fictitious Law Enforcement Character \" Archibald Barclay Willoby \" Better Known\", \"answer\":0, \"answers\":[\"PC 49\"], \"source\":\"\"},
    ]""", **parameters)
    print(f"Response from Model: {response.text}")


    # Then please add plausible but wrong answers to the `answers` array, such that answer[0] is the correct one. Finally, p
    # for line in lines:
    #     json_object = parse_json_line(line)
    #     if json_object is None:
    #         print(line)


if __name__ == "__main__":
    main()
