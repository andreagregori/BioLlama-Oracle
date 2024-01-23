from llama2_agent import Agent
from prompting import format_search_query_template


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
    agent.rag_with_pubmed(question)


test1()
