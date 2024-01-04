import textwrap


def format_text(text, width=130):
    res = textwrap.fill(text, width=width)
    return res
