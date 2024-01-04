from Bio import Entrez
from Bio import Medline

Entrez.email = "andrea.gregori@studenti.unipd.it"     # Mandatory


def search_articles(query, database='pubmed', max_results=5, sort_method='relevance') -> list[str]:
    """
    Function to search in the NCBI's databases
    :param query: string containing what to search
    :param database: name of database
    :param max_results: maximum number of result
    :param sort_method: sort order ('pub_date', 'relevance', 'Author', 'JournalName')
    :return: list of PMIDs
    """
    handle = Entrez.esearch(db=database, term=query, retmax=max_results, sort=sort_method)
    record = Entrez.read(handle)
    handle.close()

    if int(record["Count"]) > 0:
        print(f"Found {record['Count']} articles for the query '{query}'")
    else:
        print(f"No articles found for the query '{query}'.")

    return record['IdList']


def summary_articles(pmid_list: list[str], database: str = 'pubmed'):
    """
    Function to do a summary of articles given their id's.
    :param pmid_list: list of id's
    :param database: name of database
    :return: a list of dictionaries containing id, title and author list.
    """
    record_list = []
    if len(pmid_list) > 0:
        pmid_str = ",".join(pmid_list)      # to create a comma-delimited list of PMIDs
        handle = Entrez.esummary(db=database, id=pmid_str, retmode="xml")
        records = Entrez.parse(handle)

        for record in records:
            d = {'id': record['Id'], 'title': record['Title'], 'authors': record['AuthorList']}
            record_list.append(d)
        handle.close()

    return record_list


def get_abstracts_from_pmids2(pmid_list, database: str = 'pubmed'):
    """
    Function to obtain the abstracts af the articles given their id's.
    :param pmid_list: list of id's
    :param database: name of database
    :return: list of abstracts
    """
    list_ab = []
    if len(pmid_list) > 0:
        pmid_str = ",".join(pmid_list)
        handle = Entrez.efetch(db=database, id=pmid_str, retmode="xml")
        records = Entrez.read(handle)
        for paper in records['PubmedArticle']:
            list_ab.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])

        handle.close()
    return list_ab


def get_abstracts_from_pmids(pmid_list, database: str = 'pubmed'):
    """
    Function to obtain the abstracts af the articles given their id's. Using rettype="medline" and retmode="text".
    :param pmid_list: list of id's
    :param database: name of database
    :return: list of abstracts
    """
    list_ab = []
    if len(pmid_list) > 0:
        pmid_str = ",".join(pmid_list)
        handle = Entrez.efetch(db=database, id=pmid_str, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        for paper in records:
            list_ab.append(paper['AB'])
        handle.close()

    return list_ab


def get_text_abstracts_from_pmids(pmid_list, database: str = 'pubmed'):
    """
    Function to obtain the text of abstracts af the articles given their id's.
    Using retmode='text' and rettype='abstract'.
    :param pmid_list: list of id's
    :param database: name of database
    """
    if len(pmid_list) > 0:
        pmid_str = ",".join(pmid_list)
        handle = Entrez.efetch(db=database, id=pmid_str, retmode='text', rettype='abstract')
        print(handle.read())

        handle.close()


def test():
    query = "type 2 diabetes treatment options"
    search_result = search_articles(query, max_results=5)
    dict_result = summary_articles(search_result)
    print(dict_result)
    li = get_abstracts_from_pmids(search_result)
    for elem in li:
        print(elem)
