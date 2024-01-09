from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser

from prompting import get_prompt_template_with_vars, format_string_list, format_string_contexts
from output_parsers import QueryPubMed
from retrievers.entrez_retriever import search_articles, get_dicts_from_pmids
from utility import format_text
import time


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
        print(f"QUERY:\n{output}\n")

        # Parsing the output
        pydantic_parser = PydanticOutputParser(pydantic_object=QueryPubMed)
        my_query = pydantic_parser.parse(output)
        self.query = my_query.query

        print('Searching for articles using query: ', self.query)
        search_result = search_articles(self.query, max_results=limit)
        list_dicts = get_dicts_from_pmids(search_result)
        return list_dicts

    def answer_from_context(self, context: str, question: str):
        prompt = get_prompt_template_with_vars('answer_using_context.txt', ["context", "question"])
        llm_chain = LLMChain(
            llm=self.agent,
            prompt=prompt,
            verbose=True
        )
        output = llm_chain.predict(context=context, question=question)
        return output

    def choose_best_articles(self, titles_list):
        prompt = get_prompt_template_with_vars('choose_best_articles.txt', ['list_titles', 'question'])
        llm_chain = LLMChain(
            llm=self.agent,
            prompt=prompt,
            verbose=True
        )
        string_list = format_string_list(titles_list)
        output = llm_chain.predict(list_titles=string_list, question=self.user_query)

        # Parsing the output
        output_parser = CommaSeparatedListOutputParser()
        ordered_list = output_parser.parse(output)
        integer_list = [int(x) for x in ordered_list]
        decremented_list = [x - 1 for x in integer_list]
        print(decremented_list)

        return decremented_list

    def rag_with_pubmed(self,
                        user_query: str,
                        n_papers: int = 10,
                        choose_abstracts='provide_contexts'):

        pubmed_dicts = self.retrieve_pubmed_data(user_query, n_papers)

        if choose_abstracts == 'llm':
            # Selecting the best articles using the titles
            best_ab = self.choose_best_articles([d['title'] for d in pubmed_dicts if 'title' in d])
            titles = [pubmed_dicts[i]['title'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            authors = [pubmed_dicts[i]['authors'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            texts = [pubmed_dicts[i]['abstract'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            contexts = format_string_contexts(titles, authors, texts)

            response = self.answer_from_context(contexts, self.user_query)
            print(response)

        elif choose_abstracts == 'provide_contexts':
            # Feed the model with the first n_papers returned by PubMed
            titles, authors, texts = get_info_from_dicts(pubmed_dicts, n_papers)
            contexts = format_string_contexts(titles, authors, texts)

            start_time = time.time()
            response = self.answer_from_context(contexts, self.user_query)
            end_time = time.time()
            print(response)
            print(f"Execution time: {end_time - start_time}")

        elif choose_abstracts == 'embeddings':
            # TODO: embeddings with chromadb or others
            pass


def get_info_from_dicts(pubmed_dicts):
    titles = [pubmed_dicts[i]['title'] for i in range(0, len(pubmed_dicts))]
    authors = [pubmed_dicts[i]['authors'] for i in range(0, len(pubmed_dicts))]
    texts = [pubmed_dicts[i]['abstract'] for i in range(0, len(pubmed_dicts))]
    return titles, authors, texts
