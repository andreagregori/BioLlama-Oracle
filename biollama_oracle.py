from llama2_agent import Agent
from prompting import format_search_query_template


agent = Agent(agent_model="llama2",
              temperature=0,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():
    agent.rag_with_pubmed('What are the treatment options for individuals diagnosed with Type 2 diabetes?', n_papers=10)
    #agent.rag_with_pubmed('What are the common risk factors for developing osteoporosis?')
    #agent.rag_with_pubmed('What are the effects on blood pressure due to air pollution?',  n_papers=10)
    #agent.rag_with_pubmed('Is the aspartame carcinogenic? Search only in recent studies.')
    #agent.choose_best_articles()


test1()
