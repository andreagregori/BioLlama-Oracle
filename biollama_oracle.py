from llama2_agent import Agent
from utility import read_json_file, count_correct_urls
import time


agent = Agent(agent_model="llama2",
              temperature=0,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():
    '''agent.rag_with_pubmed('What are the treatment options for individuals diagnosed with Type 2 diabetes?', n_papers=5)
    agent.rag_with_pubmed('Sitting for a long time is dangerous?', n_papers=5)
    agent.chain_of_notes('What are the treatment options for individuals diagnosed with Type 2 diabetes?', n_papers=5)
    agent.rag_with_pubmed('What are the common risk factors for developing osteoporosis?')
    agent.rag_with_pubmed('What are the effects on blood pressure due to air pollution?',  n_papers=10)
    agent.rag_with_pubmed('Is the aspartame carcinogenic? Search only in recent studies.')

    agent.get_sub_queries('How does the storage and handling of vaccines affect their efficacy?')
    agent.run_chain_test('What are the common risk factors for developing osteoporosis?')

    response = agent.rag_with_pubmed('Is Rheumatoid Arthritis more common in men or women?')
    agent.self_reflect(response)
    agent.get_sub_queries('What does an abscess look like?')'''

    question = 'Is anaphylaxis a results of mast cell activation?'
    #agent.get_query_json(question)
    #agent.get_query_few_shot(question)
    #agent.rag_with_pubmed(question)
    #agent.rag_with_pubmed(question, sub_queries=True)
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    #agent.chain_of_notes(question)
    #print(agent.run_prompt(question))
    agent.rag_with_pubmed(question, sub_queries=True)


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
