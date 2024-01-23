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


def get_dicts_from_pmids(pmid_list, database: str = 'pubmed'):
    """
    Function to obtain the dicts containing info af the articles given their id's.
    Using rettype="medline" and retmode="text".
    :param pmid_list: list of id's
    :param database: name of database
    :return: list of dicts containing id, title, authors and abstract
    """
    list_dict = []
    if len(pmid_list) > 0:
        pmid_str = ",".join(pmid_list)
        handle = Entrez.efetch(db=database, id=pmid_str, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        for paper in records:
            try:
                d = {'id': paper['PMID'], 'title': paper['TI'], 'authors': paper['AU'], 'abstract': paper['AB']}
            except KeyError:
                print(f"KeyError for {paper['PMID']}")
            else:
                list_dict.append(d)
        handle.close()

    return list_dict


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


def get_urls_from_pmids(list_pmids: list[str]):
    base_url = 'http://www.ncbi.nlm.nih.gov/pubmed/'
    urls = []
    for id in list_pmids:
        urls.append(base_url + id)
    print(urls)
    return urls


def test():
    query = "type 2 diabetes treatment options"
    search_result = search_articles(query, max_results=10)
    dict_result = summary_articles(search_result)
    for i, elem in enumerate(dict_result):
        print(f"{i+1}. {elem['title']}")

