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


def write_question_to_text_file(question, filename):
    try:
        with open(filename, 'a') as file:  # 'a' appends to the file
            file.write(f"{question}\n\n")
    except Exception as e:
        print(f"Error: {e}")


def write_json_file(file_path, data):
    """
    Writes a JSON file with the provided data.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"JSON file '{file_path}' successfully created.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents as a Python object.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_questions_from_file(file_path):
    """
    Reads a text file containing questions, where each question is on a separate line,
    and questions are separated by blank lines.
    """
    questions = []
    try:
        with open(file_path, 'r') as file:
            current_question = ""
            for line in file:
                line = line.strip()  # Remove leading and trailing whitespaces
                if line:  # If the line is not blank
                    current_question += line + " "
                else:
                    if current_question:
                        questions.append(current_question.strip())
                        current_question = ""

            # Append the last question if there is no blank line after it
            if current_question:
                questions.append(current_question.strip())

        return questions
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def select_questions_BioASQ(questions, dataset_path):
    dataset = read_json_file(dataset_path)
    dataset = dataset['questions']
    data = []
    for q in dataset:
        if q['body'] in questions:
            print(q['body'])
            data.append(q)
    return data


def count_correct_urls(gt_utls, urls):
    count = 0
    for u in urls:
        if u in gt_utls:
            count += 1
    return count


'''questions = read_questions_from_file('../questions1_BioASQ.txt')
data = select_questions_BioASQ(questions, '../datasets/BioASQ/Task10BGoldenEnriched/10B1_golden.json')
data.extend(select_questions_BioASQ(questions, '../datasets/BioASQ/Task11BGoldenEnriched/11B1_golden.json'))
print(len(data))
write_json_file('../questions1_BioASQ.json', data)'''
