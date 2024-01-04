import importlib
from langchain.prompts import PromptTemplate


B_S, E_S = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_template_from_txt(name_template: str):
    text = importlib.resources.files('prompt_templates').joinpath(name_template).read_text()
    return text


def get_prompt_template_with_vars(template_name: str, input_variables: list[str] = []):
    template = get_template_from_txt(template_name)
    prompt = PromptTemplate(
        input_variables=input_variables, template=template
    )
    return prompt


def format_search_query_template(prompt: str) -> str:
    text = importlib.resources.files('prompt_templates').joinpath('query_pubmed.txt').read_text()
    text = text.format(prompt=prompt)
    return text


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_S + B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def get_prompt_with_history(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


