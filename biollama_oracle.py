from llama2_agent import Agent
from utility import read_json_file, count_correct_urls, print_articles
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
    articles = agent.interleaves_chain_of_thought(question)
    # print_articles(articles)
    agent.chain_of_thoughts(articles)


def test_on_bio_asq():
    dataset = read_json_file('../datasets/questions1_BioASQ.json')
    start_dataset = time.time()
    for q in dataset:
        print(f"Processing: {q['body']} ...")
        try:
            with open('outputs/rag_query_and_medcpt.txt', 'a') as file:
                file.write(f"Q: {q['body']}\n\n")
                start_t = time.time()
                response, urls = agent.rag_with_pubmed(q['body'], query_and_medcpt=True)
                end_t = time.time()
                print(f"Execution time: {end_t - start_t}")
                file.write('A: ' + response)
                gt_num_urls = len(q['documents'])
                correct_urls = count_correct_urls(q['documents'], urls)
                file.write(f"\n\nIDEAL: {q['ideal_answer']}")
                file.write(f"\n\nARTICLES: {len(urls)} articles found, "
                           f"{correct_urls}/{gt_num_urls} are in the ground truth.\n\n-----------\n\n")
        except Exception as e:
            print(f"Error: {e}")
    end_dataset = time.time()
    print(f"Total execution time: {end_dataset - start_dataset}")


test1()
