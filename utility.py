import textwrap
import json


def format_text(text, width=130):
    res = textwrap.fill(text, width=width)
    return res


def print_dict_structure(d, indent=2):
    formatted_json = json.dumps(d, indent=indent)
    print(formatted_json)


def write_qa_to_text_file(question, answer, filename):
    try:
        with open(filename, 'a') as file:  # 'a' appends to the file
            file.write(f"Q: {question}\nA: {answer}\n\n")
        print(f"Question and answer have been successfully written to {filename}")
    except Exception as e:
        print(f"Error: {e}")
