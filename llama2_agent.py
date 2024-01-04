from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from prompting import get_prompt_template_with_vars
from output_parsers import QueryPubMed
from retrievers.entrez_retriever import search_articles, get_abstracts_from_pmids, summary_articles
from utility import format_text


class Agent:
    def __init__(self, agent_model: str = "llama2", **kwargs):
        self.agent = Ollama(model=agent_model, **kwargs)
        self.user_query = None
        self.query = None

    def run_prompt(self, prompt):
        return self.agent(prompt)

    def retrieve_pubmed_data(self, user_query: str, limit: int = 5):
        self.user_query = user_query
        prompt = get_prompt_template_with_vars('query_pubmed_json2.txt', ['prompt', 'format_instructions'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt)
        format_instructions = """{\n"query": "put the query here",\n"description": "put the description here"\n}"""
        output = llm_chain.predict(prompt=user_query, format_instructions=format_instructions)
        #print(f"QUERY:\n{output}\n")

        # Parsing the output
        pydantic_parser = PydanticOutputParser(pydantic_object=QueryPubMed)
        my_query = pydantic_parser.parse(output)
        self.query = my_query.query

        print('Searching for pages using query: ', self.query)
        search_result = search_articles(self.query, max_results=limit)
        dict_summary = summary_articles(search_result)
        list_ab = get_abstracts_from_pmids(search_result)
        return list_ab, dict_summary

    def answer_from_context(self, context: str, question: str):
        prompt = get_prompt_template_with_vars('answer_using_context.txt', ["context", "question"])
        #print(prompt.template)
        llm_chain = LLMChain(
            llm=self.agent,
            prompt=prompt,
        )
        output = llm_chain.predict(context=context, question=question)
        return output

    def rag_with_pubmed(self,
                        user_query: str,
                        n_papers: int = 5):

        pubmed_abs, pubmed_dict = self.retrieve_pubmed_data(user_query, n_papers)
        idx = 0
        print(f"CONTEXT from '{pubmed_dict[idx]['title']}':\n {format_text(pubmed_abs[idx])}\n")
        response = self.answer_from_context(pubmed_abs[idx], self.user_query)
        print(response)

