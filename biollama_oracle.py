from llama2_agent import Agent
from llama2_agent import get_info_from_dicts
from utility import read_json_file, count_correct_urls, print_articles, get_ids_from_urls, write_json_file, get_text_snippets_bioasq, get_contexts_list, compute_means
from retrievers.entrez_retriever import get_dicts_from_pmids
from eval.retrieval_eval import *
import time
import pandas as pd
from datasets import Dataset
from datasets import load_from_disk


agent = Agent(agent_model="llama2",
              temperature=0,
              chunk_start=0,
              chunk_end=37,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():

    question = ' Do RNA binding Proteins  that bind to adenine uridine (AU)-rich elements (AREs) in the 5 untranslated region (UTR) of mRNAs (AU-RBPs) regulate the DNA Damage Response?'
    #agent.get_query_json(question)
    #agent.get_query_few_shot(question)
    #agent.rag_with_pubmed(question)
    #agent.rag_with_pubmed(question, sub_queries=True)
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    agent.retrieve_articles_pubmed(question, n_papers=10, json=False)
    #agent.answer_from_context(verbose=True)
    #agent.chain_of_notes(verbose=True)
    #agent.interleaves_chain_of_thought(n_steps=3)


def test_on_bio_asq_to_json(output_path: str):
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

        agent.retrieve_articles_pubmed(q['body'], n_papers=10, med_cpt=True)
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
                #'used_queries': agent.queries,       # remove this for MedCPT
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
        dataset.save_to_disk('outputs/med_CPT_1')        # change name year
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Total execution time: {end_dataset - start_dataset}")


def test_pubmed_query():
    question1 = 'Which disease is caused by mutations in the gene PRF1?'
    question2 = 'What protein is encoded by the GRN gene?'

    question = question1

    #Prompt #1
    agent.get_query_test(question, 'query_pubmed2.txt', 'question')
    print('----------\n')

    #Prompt #2
    agent.get_query_few_shot(question)
    print('----------\n')

    #Prompt #3
    agent.get_query_json(question)
    print('----------\n')

    #Prompt #4
    agent.get_sub_questions(question)



file_name = 'med_CPT_1.json'
test_on_bio_asq_to_json('outputs/' + file_name)

results = read_json_file('outputs/' + file_name)
mean_precision, mean_recall, mean_f1, mean_time = compute_means(results)
print('Recall: ' + str(mean_recall))
print('Precision: ' + str(mean_precision))
print('F1-score: ' + str(mean_f1))
print('Mean time: ' + str(mean_time))

#loaded_dataset = load_from_disk('outputs/answer_from_context')
#df = loaded_dataset.to_pandas().to_csv('outputs/answer_from_context.csv')

#test1()


