import textwrap
import json


def format_text(text, width=130):
    res = textwrap.fill(text, width=width)
    return res


def print_dict_structure(d, indent=2):
    formatted_json = json.dumps(d, indent=indent)
    print(formatted_json)