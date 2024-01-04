from langchain.retrievers import PubMedRetriever
from llama_index import download_loader


def get_documents_from_query(query: str):
    retriever = PubMedRetriever()
    docs = retriever.get_relevant_documents(query, retmax=5)
    print(f"Number of documents found: {len(docs)}")
    for doc in docs:
        print(doc.metadata)


def get_documents_from_query2(query: str):
    PubmedReader = download_loader("PubmedReader")
    loader = PubmedReader()
    docs = loader.load_data(search_query=query, max_results=3)
    for doc in docs:
        print(doc.metadata)


#get_documents_from_query("type 2 diabetes treatment options")
