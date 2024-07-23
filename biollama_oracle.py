from llama2_agent import Agent
from llama2_agent import get_info_from_dicts
from utility import read_json_file, count_correct_urls, print_articles, get_ids_from_urls, write_json_file, get_text_snippets_bioasq, get_contexts_list, compute_means
from retrievers.entrez_retriever import get_dicts_from_pmids
from eval.retrieval_eval import *
import time
import pandas as pd
from datasets import Dataset
from datasets import load_from_disk
from statistics import mean


agent = Agent(agent_model="llama2",
              temperature=0,
              chunk_start=0,
              chunk_end=37,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():
    question = 'What are the phenotypes linked to pathogenic variants in the CCDC88C gene?'
    agent.retrieve_articles_pubmed(question, n_papers=10)
    res = agent.answer_from_context(verbose=True)
    print(res)


def test_on_bio_asq_to_json(output_path: str):
    """
    Function to test both retrieval end generation phase
    """
    dataset = read_json_file('../datasets/questions1_BioASQ.json')
    start_dataset = time.time()

    data_res = []       # for the JSON dataset

    # For the RAGAS dataset
    questions = []      
    answers = []
    ground_truth_list = []      # the ground truth answer to the questions.
    contexts_list = []          # the contexts which were passed into the LLM to answer the question.
    
    i = 0;
    for q in dataset:
        print("\n" + str(i+1) + f") Processing: {q['body']} ...")
        start_t = time.time()

        agent.retrieve_articles_pubmed(q['body'], n_papers=10)
        response = agent.answer_from_context()      # change this to test other methods
        
        end_t = time.time()
        time_ex = round(end_t - start_t, 2)
        #print(f"Execution time: {end_t - start_t}")

        # Retrieval evaluation
        gt_pmids = get_ids_from_urls(q['documents'])
        gt_articles = get_dicts_from_pmids(gt_pmids)
        gt_context = get_text_snippets_bioasq(q)
        actual_pmids, *_ = get_info_from_dicts(agent.articles)
        recall = recall_at_k(gt_pmids, actual_pmids)
        precision = precision_at_k(gt_pmids, actual_pmids)
        f1 = f1_at_k(gt_pmids, actual_pmids)

        # Removing the authors
        retrieved_articles = agent.articles
        for d in retrieved_articles:
            if 'authors' in d:
                del d['authors']
        for d in gt_articles:
            if 'authors' in d:
                del d['authors']

        item = {'question': q['body'], 'answer': response, 'time_ex': time_ex,
                'gt_answer': q['ideal_answer'], 'gt_articles': gt_articles, 'gt_context': gt_context,
                'retrieved_articles': retrieved_articles,
                'used_queries': agent.queries,       # remove this for MedCPT
                'recall': recall, 'precision': precision, 'f1': f1}
        data_res.append(item)

        ### CREATING RAGAS DATASET ###
        questions.append(q['body'])
        answers.append(response)
        ground_truth_list.append(q['ideal_answer'][0])      # might be more than one ideal answer, we take always the first
        contexts_list.append(get_contexts_list(retrieved_articles))
        i+=1

    end_dataset = time.time()
    data = {'question': questions, 'answer': answers, 'contexts': contexts_list, 'ground_truth': ground_truth_list}

    try:
        write_json_file(output_path, data_res)
        dataset = Dataset.from_dict(data)
        dataset.save_to_disk('outputs/CoT_1')        # change name year
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Total execution time: {end_dataset - start_dataset}")


def test_retrieval_to_json(output_path: str):
    """
    Function to test only the retrieval phase
    """
    dataset = read_json_file('../datasets/questions1_BioASQ.json')

    data_res = []       # for the JSON dataset
    
    i = 0;
    for q in dataset:
        print("\n" + str(i+1) + f") Processing: {q['body']} ...")

        agent.retrieve_articles_pubmed(q['body'], n_papers=10, temp=50)    ### CHANGE THIS ###
        #agent.interleaves_chain_of_thought(n_steps=3)

        # Retrieval evaluation
        gt_pmids = get_ids_from_urls(q['documents'])
        gt_articles = get_dicts_from_pmids(gt_pmids)
        gt_context = get_text_snippets_bioasq(q)
        actual_pmids, *_ = get_info_from_dicts(agent.articles)
        recall = recall_at_k(gt_pmids, actual_pmids)
        precision = precision_at_k(gt_pmids, actual_pmids)
        f1 = f1_at_k(gt_pmids, actual_pmids)

        # Removing the authors
        retrieved_articles = agent.articles
        for d in retrieved_articles:
            if 'authors' in d:
                del d['authors']
        for d in gt_articles:
            if 'authors' in d:
                del d['authors']

        item = {'question': q['body'],
                'gt_answer': q['ideal_answer'], 'gt_articles': gt_articles, 'gt_context': gt_context,
                'retrieved_articles': retrieved_articles,
                'used_queries': agent.queries,       # remove this for MedCPT
                'recall': recall, 'precision': precision, 'f1': f1}
        data_res.append(item)
        i+=1

    try:
        write_json_file(output_path, data_res)
    except Exception as e:
        print(f"Error: {e}")

    # Computing means
    precisions = [d['precision'] if d['precision'] is not None else 0 for d in data_res]
    recalls = [d['recall'] if d['recall'] is not None else 0 for d in data_res]
    f1_scores = [d['f1'] if d['f1'] is not None else 0 for d in data_res]
    mean_precision = round(mean(precisions), 4)
    mean_recall = round(mean(recalls), 4)
    mean_f1 = round(mean(f1_scores), 4)

    print('Recall: ' + str(mean_recall))
    print('Precision: ' + str(mean_precision))
    print('F1-score: ' + str(mean_f1))


def test_pubmed_query():
    """
    Function to test the methods to generate the query
    """
    question1 = 'Which disease is caused by mutations in the gene PRF1?'
    question2 = 'What protein is encoded by the GRN gene?'

    question = question1

    #Prompt #1
    agent.get_query_test(question, 'query_pubmed2.txt', 'question')
    print('----------\n')
    '''
    #Prompt #2
    agent.get_query_few_shot(question)
    print('----------\n')

    #Prompt #3
    agent.get_query_json(question)
    print('----------\n')

    #Prompt #4
    agent.get_sub_questions(question)
    '''


def test_on_custom_dataset(dataset_path, output_path):
    """
    Function to test both retrieval end generation phase on the custom dataset
    """
    dataset = read_json_file(dataset_path)
    start_dataset = time.time()

    data_res = []       # for the JSON dataset

    i = 0;
    for q in dataset:
        print("\n" + str(i+1) + f") Processing: {q['question']} ...")
        start_t = time.time()

        agent.retrieve_articles_pubmed(q['question'], n_papers=10)
        std_response = agent.answer_from_context()
        cot_response = agent.chain_of_thoughts()
        con_response = agent.chain_of_notes()
        
        end_t = time.time()
        time_ex = round(end_t - start_t, 2)

        # Retrieval evaluation
        gt_pmids = q['pmids']
        gt_articles = get_dicts_from_pmids(gt_pmids)
        gt_context = get_contexts_list(gt_articles)
        actual_pmids, *_ = get_info_from_dicts(agent.articles)
        recall = recall_at_k(gt_pmids, actual_pmids)
        precision = precision_at_k(gt_pmids, actual_pmids)
        f1 = f1_at_k(gt_pmids, actual_pmids)

        # Removing the authors
        retrieved_articles = agent.articles
        for d in retrieved_articles:
            if 'authors' in d:
                del d['authors']
        for d in gt_articles:
            if 'authors' in d:
                del d['authors']

        item = {'report_id': q['report'], 'question': q['question'],
                'std_response': std_response, 'cot_response': cot_response, 'con_response': con_response,
                'gt_answer': q['gt_answer'], 'gt_risposta': q['gt_risposta'],
                'gt_articles': gt_articles, 'gt_context': gt_context,
                'retrieved_articles': retrieved_articles,
                'used_queries': agent.queries,       # remove this for MedCPT
                'recall': recall, 'precision': precision, 'f1': f1}
        data_res.append(item)

        i+=1

    end_dataset = time.time()

    # Computing means 
    precisions = [d['precision'] if d['precision'] is not None else 0 for d in data_res]
    recalls = [d['recall'] if d['recall'] is not None else 0 for d in data_res]
    f1_scores = [d['f1'] if d['f1'] is not None else 0 for d in data_res]
    mean_precision = round(mean(precisions), 4)
    mean_recall = round(mean(recalls), 4)
    mean_f1 = round(mean(f1_scores), 4)

    print('Recall: ' + str(mean_recall))
    print('Precision: ' + str(mean_precision))
    print('F1-score: ' + str(mean_f1))

    try:
        write_json_file(output_path, data_res)
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Total execution time: {end_dataset - start_dataset}")




file_name = 'test_1.json'

path1 = 'outputs/' + file_name
path2 = 'outputs/01-only_retrieval/' + file_name
#test_on_bio_asq_to_json(path1)
test_retrieval_to_json(path2)

'''
results = read_json_file(path1)
mean_precision, mean_recall, mean_f1, mean_time = compute_means(results)
print('Recall: ' + str(mean_recall))
print('Precision: ' + str(mean_precision))
print('F1-score: ' + str(mean_f1))
print('Mean time: ' + str(mean_time))
'''

#loaded_dataset = load_from_disk('outputs/answer_from_context')
#df = loaded_dataset.to_pandas().to_csv('outputs/answer_from_context.csv')

#test_on_custom_dataset('../datasets/custom_dataset.json', 'outputs/03-custom_dataset/' + file_name)
#test1()




