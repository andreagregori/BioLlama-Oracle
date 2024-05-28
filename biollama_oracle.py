from llama2_agent import Agent
from llama2_agent import get_info_from_dicts
from utility import read_json_file, count_correct_urls, print_articles, get_ids_from_urls, write_json_file
from retrievers.entrez_retriever import get_dicts_from_pmids
from eval.retrieval_eval import *
import time


agent = Agent(agent_model="llama2",
              temperature=0,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():

    question = 'Do mutations in KCNT2 only cause phenotypes with epilepsy?'
    #agent.get_query_json(question)
    #agent.get_query_few_shot(question)
    #agent.rag_with_pubmed(question)
    #agent.rag_with_pubmed(question, sub_queries=True)
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    agent.retrieve_articles_pubmed(question, n_papers=5, med_cpt=True)
    articles = agent.interleaves_chain_of_thought(verbose=False)
    print_articles(articles)


def test_on_bio_asq_to_json(output_path: str):
    dataset = read_json_file('../datasets/questions1_BioASQ.json')
    start_dataset = time.time()
    data_res = []
    for q in dataset:
        print(f"Processing: {q['body']} ...")
        
        start_t = time.time()
        agent.retrieve_articles_pubmed(q['body'])

        response = agent.answer_from_context()      # change this to test other methods
        
        end_t = time.time()
        print(f"Execution time: {end_t - start_t}")

        # Retrieval evaluation
        gt_pmids = get_ids_from_urls(q['documents'])
        gt_articles = get_dicts_from_pmids(gt_pmids)
        actual_pmids, *_ = get_info_from_dicts(agent.articles)
        recall = recall_at_k(gt_pmids, actual_pmids)
        precision = precision_at_k(gt_pmids, actual_pmids)
        f1 = f1_at_k(gt_pmids, actual_pmids)

        # Generation evaluation
        # TODO: try ragas framework

        item = {'question': q['body'], 'answer': response, 'ideal_answer': q['ideal_answer'],
                'gt_articles': gt_articles, 'retrieved_articles': agent.articles,
                'recall': recall, 'precision': precision, 'f1': f1}
        data_res.append(item)

    try:
        print(type(data_res))
        write_json_file(output_path, data_res)
    except Exception as e:
        print(f"Error: {e}")
    end_dataset = time.time()
    print(f"Total execution time: {end_dataset - start_dataset}")


#test_on_bio_asq_to_json('outputs/rag.json')
test1()
