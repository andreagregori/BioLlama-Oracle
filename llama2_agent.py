from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser

from prompting import get_prompt_template_with_vars, format_string_list, format_string_contexts
from output_parsers import QueryPubMed, get_list_from_text
from retrievers.entrez_retriever import search_articles, get_dicts_from_pmids, get_urls_from_pmids
from ollama_requests import generate_response
from retrievers.medCPT_retriever import MedCPT, MedCPTCrossEncoder
import time
import numpy as np


class Agent:
    def __init__(self, agent_model: str = "llama2", chunk_start=30, chunk_end=36, **kwargs):
        self.agent = Ollama(model=agent_model, **kwargs)    # not using
        self.med_cpt_retriever = MedCPT(chunk_start, chunk_end)
        self.med_cpt_cross_enc = MedCPTCrossEncoder()
        self.user_query = None
        self.query = None
        self.articles = None

    def run_prompt(self, prompt):
        return self.agent(prompt)

    def run_chain_test(self, question):
        prompt = get_prompt_template_with_vars('query_few_shot.txt', ['prompt'])
        llm_chain = LLMChain(llm=self.agent, prompt=prompt, verbose=False)
        output = llm_chain.predict(prompt=question)
        print(output)

    def get_query_few_shot(self, user_query: str) -> str:
        """
        Returns a query prompting the LLM with few-shot examples. Before 

        Args:
            user_query (str): the user question.

        Returns:
            str: the query.
        """
        #self.user_query = user_query
        template = get_prompt_template_with_vars('query_few_shot2.txt', ['prompt'])
        prompt = template.format(prompt=user_query)
        output = generate_response(prompt)
        response = output['response']
        query = response.split("\n")[0]
        self.query = query
        return query

    def get_query_json(self, user_query: str) -> str:
        """
        Returns a query using the LLM to obtain the query as a json object.

        Args:
            user_query (str): the user question.

        Returns:
            str: the query.
        """
        # self.user_query = user_query
        template = get_prompt_template_with_vars('query_pubmed_json2.txt', ['prompt', 'format_instructions'])
        format_instructions = """{\n"query": "put the query here",\n"description": "put the description here"\n}"""
        prompt = template.format(prompt=user_query, format_instructions=format_instructions)
        output = generate_response(prompt)
        response = output['response']
        print(f"QUERY:\n{response}\n")

        # Parsing the output
        pydantic_parser = PydanticOutputParser(pydantic_object=QueryPubMed)
        my_query = pydantic_parser.parse(response)
        self.query = my_query.query
        return my_query.query

    def get_sub_questions(self, user_query: str) -> list[str]:
        """
        Returns a list of sub-questions related to the user question.

        Args:
            user_query (str): the user question.

        Returns:
            list[str]: list of sub-questions related to the user question.
        """
        template = get_prompt_template_with_vars('generate_sub_queries.txt', ['question'])
        prompt = template.format(question=user_query)
        output = generate_response(prompt)
        response = output['response']
        sub_questions = get_list_from_text(response)
        print(sub_questions)
        return sub_questions

    def get_sub_queries(self, sub_questions: list[str], json=False) -> list[str]:
        """
        Returns a list of sub-queries related to the sub-questions.

        Args:
            sub_questions (list[str]): list of sub-questions.
            json (bool, optional): if True, the LLM will output valid json containing the query. Defaults to False.

        Returns:
            list[str]: list of sub-queries.
        """
        sub_queries = []
        for q in sub_questions:
            if json:
                sub_queries.append(self.get_query_json(q).lower())
            else:
                sub_queries.append(self.get_query_few_shot(q).lower())
        sub_queries = list(set(sub_queries))
        print(sub_queries)
        return sub_queries
    
    def retrieve_articles_pubmed(self,
                                user_query: str,
                                n_papers: int = 10,
                                choose_abstracts='provide_contexts',
                                med_cpt=False,
                                query_and_medcpt=False,
                                json=False,
                                sub_queries=False) -> list[dict]:
        """
        Retrieves articles from PubMed in different ways.

        Args:
            user_query (str): the user question
            n_papers (int, optional): maximum number of articles to retrieve. Defaults to 10.
            choose_abstracts (str, optional): way to choose the articles (values are 'llm', 'provide_contexts' and 'embeddings'). Defaults to 'provide_contexts'.
            med_cpt (bool, optional): if True MedCPT is used as retriever. Defaults to False.
            query_and_medcpt (bool, optional): if True, MedCPT is used as retriever with a query generated by the LLM. Defaults to False.
            json (bool, optional): if True, the LLM will output valid json containing the query. Defaults to False.
            sub_queries (bool, optional): if True, the LLM will decompose the user query in sub-queries. Defaults to False.

        Returns:
            list[dict]: a list of dicts containing al the info of the articles.
        """
        self.user_query = user_query
        if med_cpt:
            # Use MedCPT as retriever
            pmids_list = self.med_cpt_retriever.retrieve_documents_pmids([user_query], n_papers)
            pubmed_dicts = get_dicts_from_pmids(pmids_list)

        elif query_and_medcpt:
            # Use query generated from LLM and MedCPT as retriever
            if json:
                query = self.get_query_json(user_query)
            else:
                query = self.get_query_few_shot(user_query)
            pmids_list = self.med_cpt_retriever.retrieve_documents_pmids([query])
            pubmed_dicts = get_dicts_from_pmids(pmids_list)

        elif sub_queries:
            # Generate sub queries to expand the search
            sub_questions = self.get_sub_questions(user_query)
            sub_queries = self.get_sub_queries(sub_questions, json)
            pubmed_dicts = self.retrieve_pubmed_multi_queries(sub_queries, limit=10)

            # Filtering the best articles using MedCPT cross encoder
            _, _, _, articles = get_info_from_dicts(pubmed_dicts)
            _, indexes = self.med_cpt_cross_enc.get_ranks_articles(user_query, articles)
            pubmed_dicts = sort_articles(pubmed_dicts, indexes)
            print(f"len: {len(pubmed_dicts)}")
            pubmed_dicts = pubmed_dicts[:10]
            print(f"len: {len(pubmed_dicts)}")
        else:
            if json:
                query = self.get_query_json(user_query)
            else:
                query = self.get_query_few_shot(user_query)
            pubmed_dicts = self.retrieve_pubmed_data(query, n_papers)

        self.articles = pubmed_dicts
        return pubmed_dicts

    def retrieve_pubmed_data(self, query: str, limit: int = 5) -> list[dict]:
        """
        Returns a list of dict containing the retrieved articles.
        Each dict represents an article and it contains: id, title, authors and abstract.

        Args:
            query (str): the query.
            limit (int, optional):  maximum number of articles to retrieve. Defaults to 5.

        Returns:
            list[dict]: list of dicts containing all the info of the articles.
        """
        #print('Searching for articles using query: ', query)
        search_result = search_articles(query, max_results=limit)
        list_dicts = get_dicts_from_pmids(search_result)
        return list_dicts

    def retrieve_pubmed_multi_queries(self, queries: list[str], limit: int = 3):
        """
        Returns a list of dict containing the retrieved articles, gathered using the list of queries.

        Args:
            queries (list[str]): list of queries.
            limit (int, optional): maximum number of articles to retrieve for each query. Defaults to 3.

        Returns:
            list: a list of dicts containing al the info of the articles.
        """
        result_list = []
        for q in queries:
            result_list.extend(search_articles(q, max_results=limit))
        result_list = list(set(result_list))
        list_dicts = get_dicts_from_pmids(result_list)
        return list_dicts


    def choose_best_articles(self, titles_list: list[str]):
        """
        Use the LLM to choose the best articles from their titles.

        Args:
            titles_list (list[str]): list of titles.

        Returns:
            list: ordered list of indexes to access the articles.
        """
        template = get_prompt_template_with_vars('choose_best_articles.txt', ['list_titles', 'question'])
        string_list = format_string_list(titles_list)
        prompt = template.format(list_titles=string_list, question=self.user_query)
        output = generate_response(prompt)

        # Parsing the output
        output_parser = CommaSeparatedListOutputParser()
        ordered_list = output_parser.parse(output['response'])
        integer_list = [int(x) for x in ordered_list]
        decremented_list = [x - 1 for x in integer_list]
        print(decremented_list)

        return decremented_list

    def rag_with_pubmed(self,
                        user_query: str,
                        n_papers: int = 10,
                        choose_abstracts='provide_contexts',
                        med_cpt=False,
                        query_and_medcpt=False,
                        json=False,
                        sub_queries=False):
        """
        TODO: da rivedere questa funzione. Potrebbe servire per scegliere tra i vari approcci.
        """
        response = None
        urls = None
        if choose_abstracts == 'llm':
            # Selecting the best articles using the titles
            best_ab = self.choose_best_articles([d['title'] for d in self.articles if 'title' in d])
            ids = [self.articles[i]['id'] for i in best_ab if 0 <= i < len(self.articles)]
            titles = [self.articles[i]['title'] for i in best_ab if 0 <= i < len(self.articles)]
            #authors = [pubmed_dicts[i]['authors'] for i in best_ab if 0 <= i < len(pubmed_dicts)]
            texts = [self.articles[i]['abstract'] for i in best_ab if 0 <= i < len(self.articles)]
            contexts = format_string_contexts(titles, texts)
            response = self.answer_from_context(contexts, self.user_query)
            urls = get_urls_from_pmids(ids)

        elif choose_abstracts == 'provide_contexts':
            # Feed the model with the first n_papers returned by PubMed
            ids, titles, authors, texts = get_info_from_dicts(self.articles)
            contexts = format_string_contexts(titles, texts)
            start_time = time.time()
            response = self.answer_from_context(contexts, self.user_query)
            end_time = time.time()
            #print(f"Execution time: {end_time - start_time}")
            urls = get_urls_from_pmids(ids)

        elif choose_abstracts == 'embeddings':
            # TODO: embeddings with chromadb or others (MedCPT)
            pass
        print(response)
        return response, urls
    
    def answer_from_context(self, verbose=False) -> str:
        """
        Generate answer from the context.

        Args:
            verbose (bool, optional):  if True, print the prompt for the LLM. Defaults to False.

        Returns:
            str: response.
        """
        _, titles, _, texts = get_info_from_dicts(self.articles)
        contexts = format_string_contexts(titles, texts)
        template = get_prompt_template_with_vars('answer_using_context.txt', ["context", "question"])
        prompt = template.format(context=contexts, question=self.user_query)
        output = generate_response(prompt, verbose=verbose)
        response = output['response']
        # print(response)
        return response

    def chain_of_notes(self, verbose=False) -> str:
        """
        Uses chain of notes on the articles stored on self.articles

        Args:
            verbose (bool, optional):  if True, print the prompt for the LLM. Defaults to False.

        Returns:
            str: the response
        """
        template = get_prompt_template_with_vars('chain_of_notes.txt', ['question', 'articles_list'])
        _, titles, _, texts = get_info_from_dicts(self.articles)
        articles_list = format_string_contexts(titles, texts)
        prompt = template.format(question=self.user_query, articles_list=articles_list)
        output = generate_response(prompt, verbose=verbose)
        response = output['response']
        print(response)
        return response

    def interleaves_chain_of_thought(self, k: int=3, n_steps: int=2, verbose=False) -> list[dict]:
        """
        Uses the articles contained in self.articles as base set. Then uses Interleaves CoT to retrieve k*n_steps new articles.

        Args:
            k (int, optional): number of articles to retrieve at each retrieve step. Defaults to 3.
            n_steps (int, optional): number of time to repeat the reason and retrieve step. Defaults to 2.
            verbose (bool, optional): if True, print the prompt for the LLM. Defaults to False.

        Returns:
            list[dict]: a list of dicts containing al the info of the articles.
        """
        template = get_prompt_template_with_vars('interleaves_CoT_reason3.txt', ['articles', 'question', 'cot_sents'])
        CoT_queries = []
        for i in range(0, n_steps):
            ids, titles, _, texts = get_info_from_dicts(self.articles)
            articles_list = format_string_contexts(titles, texts)
            cot_sents = ', '.join(map(str, CoT_queries))
            prompt = template.format(articles=articles_list, question=self.user_query, cot_sents=cot_sents)
            output = generate_response(prompt, verbose=verbose)
            response = output['response']
            new_query = response.split("\n")[0]
            CoT_queries.append(new_query)
            print("New query: " + new_query)

            # Retrieving new articles
            temp = self.retrieve_pubmed_data(new_query, k)
            if len(temp) == 0:  # Stop if there are no more new articles
                print("Stop IRCoT")
                break
            for elem in temp:   # ignoring duplicated articles
                if elem not in self.articles:
                    self.articles.append(elem)
                else:
                    print("Duplicated article")
        
        return self.articles
    

    def chain_of_thoughts(self, verbose=False):
        """
        Answer the question using the articles reasoning step by step.

        Args:
            verbose (bool, optional): if True print the prompt for the LLM. Defaults to False.

        Returns:
            str: the response.
        """
        _, titles, _, texts = get_info_from_dicts(self.articles)
        context = format_string_contexts(titles, texts)
        template = get_prompt_template_with_vars('answer_using_CoT.txt', ['context', 'question'])
        prompt = template.format(context=context, question=self.user_query)
        output = generate_response(prompt, verbose=verbose)
        response = output['response']
        print(response)
        return response    


    def self_reflect(self, answer: str, verbose=False) -> str:
        """
        Uses the LLM to check if the given answer is sufficient.

        Args:
            answer (str): the answer to check.
            verbose (bool, optional): if True print the prompt for the LLM. Defaults to False.

        Returns:
            str: the response
        """
        template = get_prompt_template_with_vars('self_reflect.txt', ['answer', 'question'])
        prompt = template.format(answer=answer, question=self.user_query)
        output = generate_response(prompt, verbose=verbose)
        response = output['response']
        print(response)
        return response


def get_info_from_dicts(pubmed_dicts: list[dict]):
    """
    Extracts ids, titles, authors and text from a list of dicts.

    Args:
        pubmed_dicts (list[dict]): a list of dicts.

    Returns:
        list[str]: four list containing ids, titles, authors and texts.
    """
    ids = [pubmed_dicts[i]['id'] for i in range(0, len(pubmed_dicts))]
    titles = [pubmed_dicts[i]['title'] for i in range(0, len(pubmed_dicts))]
    authors = [pubmed_dicts[i]['authors'] for i in range(0, len(pubmed_dicts))]
    texts = [pubmed_dicts[i]['abstract'] for i in range(0, len(pubmed_dicts))]
    return ids, titles, authors, texts


def sort_articles(articles, indices_tensor):
    sorted_articles = np.array(articles)[indices_tensor]
    sorted_articles_list = sorted_articles.tolist()
    return sorted_articles_list
