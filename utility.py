import textwrap
import json
import csv
from statistics import mean


def format_text(text, width=130):
    res = textwrap.fill(text, width=width)
    return res


def print_dict_structure(d, indent=2):
    formatted_json = json.dumps(d, indent=indent)
    print(formatted_json)


def print_articles(list_dicts):
    articles_string = ""
    for paper in list_dicts:
        articles_string += f"{paper['id']}\nArticle title: {paper['title']}\n{paper['abstract']}\n\n"
    print(articles_string)


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


def count_correct_urls(gt_urls, urls):
    count = 0
    for u in urls:
        if u in gt_urls:
            count += 1
    return count


def get_ids_from_urls(documents: list[str]) -> list[str]:
    """
    Returns the pmids inside the given list of urls.

    Args:
        documents (list[str]): list of urls.

    Returns:
        list[str]: list of pmids.
    """
    ids = []
    for url in documents:
        id = url.split("/")[-1]
        ids.append(id)
    return ids

def get_text_snippets_bioasq(item):
    gt_context = []
    for elem in item['snippets']:
        gt_context.append(elem['text'])

    return gt_context

def get_contexts_list(articles):
    context_list = []
    for a in articles:
        context_list.append(a['title'])
        if a['abstract'] != 'Abstract not available.':
            context_list.append(a['abstract'])
    return context_list

def compute_means(results):
    precisions = [d['precision'] if d['precision'] is not None else 0 for d in results]
    recalls = [d['recall'] if d['recall'] is not None else 0 for d in results]
    f1_scores = [d['f1'] if d['f1'] is not None else 0 for d in results]
    times = [d['time_ex'] if d['time_ex'] is not None else 0 for d in results]

    mean_precision = round(mean(precisions), 4)
    mean_recall = round(mean(recalls), 4)
    mean_f1 = round(mean(f1_scores), 4)
    mean_time = round(mean(times), 2)
    return mean_precision, mean_recall, mean_f1, mean_time


def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV file and handle BOM
    with open(csv_file_path, mode='r', newline='', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
    
    # Convert to JSON
    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    
    # Save the JSON data to a file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json_file.write(json_data)
    
    return json_file_path



'''questions = read_questions_from_file('../questions1_BioASQ.txt')
data = select_questions_BioASQ(questions, '../datasets/BioASQ/Task10BGoldenEnriched/10B1_golden.json')
data.extend(select_questions_BioASQ(questions, '../datasets/BioASQ/Task11BGoldenEnriched/11B1_golden.json'))
print(len(data))
write_json_file('../questions1_BioASQ.json', data)'''