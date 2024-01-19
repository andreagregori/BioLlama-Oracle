from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser

from prompting import get_prompt_template_with_vars, format_string_list, format_string_contexts
from output_parsers import QueryPubMed, get_list_from_text
from retrievers.entrez_retriever import search_articles, get_dicts_from_pmids
from utility import write_qa_to_text_file
import time


class Agent:
    def __init__(self, agent_model: str = "llama2", **kwargs):
        self.agent = Ollama(model=agent_model, **kwargs)
        self.user_query = None
        self.query = None

    def run_prompt(self, prompt):
        return self.agent(prompt)

    def run_chain_test(self, question):
        prompt = get_prompt_template_with_vars('query_few_shot.txt', ['prompt'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt, verbose=False)
        output = llm_chain.predict(prompt=question)
        print(output)

    def get_query_few_shot(self, user_query: str):
        """
        Returns a query prompting the LLM with few-shot examples.
        """
        #self.user_query = user_query
        prompt = get_prompt_template_with_vars('query_few_shot.txt', ['prompt'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt, verbose=False)
        output = llm_chain.predict(prompt=user_query)
        query = output.split("\n")[0]
        print(query)
        self.query = query
        return query

    def get_query_json(self, user_query: str):
        """
        Returns a query using the LLM to obtain the query as a json object.
        """
        #self.user_query = user_query
        prompt = get_prompt_template_with_vars('query_pubmed_json2.txt', ['prompt', 'format_instructions'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt)
        format_instructions = """{\n"query": "put the query here",\n"description": "put the description here"\n}"""
        output = llm_chain.predict(prompt=user_query, format_instructions=format_instructions)
        #print(f"QUERY:\n{output}\n")

        # Parsing the output
        pydantic_parser = PydanticOutputParser(pydantic_object=QueryPubMed)
        my_query = pydantic_parser.parse(output)
        self.query = my_query.query
        return my_query.query

    def get_sub_questions(self, user_query: str):
        """
        Returns a list of sub-questions related to the user question.
        """
        prompt = get_prompt_template_with_vars('generate_sub_queries.txt', ['question'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt, verbose=True)
        output = llm_chain.predict(question=user_query)
        sub_questions = get_list_from_text(output)
        print(sub_questions)
        return sub_questions

    def get_sub_queries(self, sub_questions):
        """
        Returns a list of sub-queries related to the sub-questions.
        """
        sub_queries = []
        for q in sub_questions:
            sub_queries.append(self.get_query_json(q))
        sub_queries = list(set(sub_queries))
        print(sub_queries)
        return sub_queries

    def retrieve_pubmed_data(self, query: str, limit: int = 5):
        """
        Returns a list of dict containing the retrieved articles.
        Each dict represents an article and it contains: id, title, authors and abstract.
        """
        #print('Searching for articles using query: ', query)
        search_result = search_articles(query, max_results=limit)
        list_dicts = get_dicts_from_pmids(search_result)
        return list_dicts

    def retrieve_pubmed_multi_queries(self, queries: list[str], limit: int = 3):
        """
        Returns a list of dict containing the retrieved articles, gathered using the list of queries.
        """
        result_list = []
        for q in queries:
            result_list.extend(search_articles(q, max_results=limit))
        result_list = list(set(result_list))
        list_dicts = get_dicts_from_pmids(result_list)
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
        """
        Use the LLM to choose the best articles from their titles.
        """
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
                        choose_abstracts='provide_contexts',
                        sub_queries=True):
        self.user_query = user_query
        if sub_queries:
            sub_questions = self.get_sub_questions(user_query)
            sub_queries = self.get_sub_queries(sub_questions)
            pubmed_dicts = self.retrieve_pubmed_multi_queries(sub_queries)
        else:
            query = self.get_query_json(user_query)
            pubmed_dicts = self.retrieve_pubmed_data(query, n_papers)

        if choose_abstracts == 'llm':
            # Selecting the best articles using the titles
            best_ab = self.choose_best_articles([d['title'] for d in pubmed_dicts if 'title' in d])
            titles = [pubmed_dicts[i]['title'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            #authors = [pubmed_dicts[i]['authors'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            texts = [pubmed_dicts[i]['abstract'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            contexts = format_string_contexts(titles, texts)

            response = self.answer_from_context(contexts, self.user_query)
            print(response)
            return response

        elif choose_abstracts == 'provide_contexts':
            # Feed the model with the first n_papers returned by PubMed
            titles, authors, texts = get_info_from_dicts(pubmed_dicts)
            contexts = format_string_contexts(titles, texts)

            start_time = time.time()
            response = self.answer_from_context(contexts, self.user_query)
            end_time = time.time()
            print(response)
            print(f"Execution time: {end_time - start_time}")
            return response

        elif choose_abstracts == 'embeddings':
            # TODO: embeddings with chromadb or others
            pass

    def chain_of_notes(self, user_query: str, n_papers: int = 5):
        self.user_query = user_query
        query = self.get_query_json(user_query)
        pubmed_dicts = self.retrieve_pubmed_data(query, n_papers)
        prompt = get_prompt_template_with_vars('chain_of_notes.txt', ['question', 'articles_list'])
        llm_chain = LLMChain(
            llm=self.agent,
            prompt=prompt,
            verbose=True
        )
        titles, _, texts = get_info_from_dicts(pubmed_dicts)
        articles_list = format_string_contexts(titles, texts)
        output = llm_chain.predict(question=self.user_query, articles_list=articles_list)
        print(output)

        return output

    def interleaves_chain_of_thought(self, user_query: str, n_papers: int = 5):
        # TODO: to finish
        pass

    def self_reflect(self, answer: str):
        prompt = get_prompt_template_with_vars('self_reflect.txt', ['answer', 'question'])
        llm_chain = LLMChain(
            llm=self.agent,
            prompt=prompt,
            verbose=True
        )
        output = llm_chain.predict(answer=answer, question=self.user_query)
        print(output)
        return output


def get_info_from_dicts(pubmed_dicts):
    titles = [pubmed_dicts[i]['title'] for i in range(0, len(pubmed_dicts))]
    authors = [pubmed_dicts[i]['authors'] for i in range(0, len(pubmed_dicts))]
    texts = [pubmed_dicts[i]['abstract'] for i in range(0, len(pubmed_dicts))]
    return titles, authors, texts
