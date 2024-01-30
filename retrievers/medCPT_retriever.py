import faiss
import torch
import numpy as np
import json
import time
from transformers import AutoTokenizer, AutoModel

faiss_dir = 'faiss_indexes/'
emb_dir = '../PubMed_embeddings/'


class MedCPT:
    def __init__(self, chunk_start=None, chunk_end=None):
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self._chunk_start = chunk_start
        self._chunk_end = chunk_end
        self.index = None
        self.pmids = None
        self.total_results = []

    @property
    def chunk_start(self):
        return self._chunk_start

    @property
    def chunk_end(self):
        return self._chunk_end

    @chunk_start.setter
    def chunk_start(self, num):
        self._chunk_start = num

    @chunk_end.setter
    def chunk_end(self, num):
        self._chunk_end = num

    def write_faiss_index(self, start, end):
        """
        Write on disk a faiss IndexFlatIP referred to chunks with number between start and end.
        """
        all_chunk_embeds = []
        for chunk_number in range(start, end + 1):
            chunk_embeds = read_embeds_chunk(chunk_number)
            all_chunk_embeds.append(chunk_embeds)

        all_chunk_embeds = np.concatenate(all_chunk_embeds, axis=0)
        self.index = faiss.IndexFlatIP(768)
        self.index.add(all_chunk_embeds)
        if start == end:
            name = f'chunk{start}FlatIP.index'
        else:
            name = f'chunk{start}_{end}FlatIP.index'
        faiss.write_index(self.index, faiss_dir + name)
        print(f"Index {name} saved")

    def read_index(self, path):
        self.index = faiss.read_index(path)        # add faiss.IO_FLAG_MMAP to read from disk

    def load_pmids(self, chunk_number):
        """
        Reads a specific pmids_chunk file.
        """
        pmids_path = emb_dir + f"pmids_chunk_{chunk_number}.json"
        chunk_pmids = json.load(open(pmids_path))
        self.pmids = chunk_pmids

    def search_queries(self, queries, k=5):
        """
        Search using the current self.index
        """
        with torch.no_grad():
            encoded = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=64,
            )

            embeds = self.model(**encoded).last_hidden_state[:, 0, :]
            scores, inds = self.index.search(embeds, k=k)

        results = []
        for idx, query in enumerate(queries):
            query_results = []
            for score, ind in zip(scores[idx], inds[idx]):
                pmid = self.pmids[ind]
                query_results.append({"PMID": pmid, "Score": score})
            results.append({"Query": query, "Results": query_results})

        return results

    def search_in_all_chunks(self, queries):
        for num in range(self.chunk_start, self.chunk_end + 1):
            print(f"Searching on chunk {num} ...")
            self.load_pmids(num)
            self.read_index(faiss_dir + f'chunk{num}FlatIP.index')
            search_results = self.search_queries(queries)
            self.total_results.extend(search_results)

            '''for result in search_results:
                print(f"Query: {result['Query']}")
                for r in result['Results']:
                    print(f"PMID: {r['PMID']}; Score: {r['Score']}")'''

    def combine_results(self, top_n=10):
        combined_scores = {}

        # Combine scores from different results
        for query_result in self.total_results:
            query = query_result["Query"]
            for result in query_result["Results"]:
                pmid = result["PMID"]
                score = result["Score"]

                if pmid not in combined_scores:
                    combined_scores[pmid] = 0
                combined_scores[pmid] += score

        # Select the top n scores
        top_pmid_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_results = [{"PMID": pmid, "Score": score} for pmid, score in top_pmid_scores]
        top_pmids = [result["PMID"] for result in top_results]

        print(f"\nTOP {top_n} RESULTS")
        for result in top_results:
            print(f"PMID: {result['PMID']}; Score: {result['Score']}")

        return top_pmids

    def retrieve_documents_pmids(self, queries):
        self.search_in_all_chunks(queries)
        tot_pmids = self.combine_results()
        return tot_pmids


def read_embeds_chunk(chunk_number):
    embeds_path = emb_dir + f"embeds_chunk_{chunk_number}.npy"
    chunk_embeds = np.load(embeds_path)
    return chunk_embeds


# Example usage:
'''chunk_start = 30
chunk_end = 36
pubmed_search = MedCPT(chunk_start, chunk_end)

queries_to_search = [
    "Is anaphylaxis a result of mast cell activation?",
]

start_time = time.time()
tot_pmids = pubmed_search.retrieve_documents_pmids(queries_to_search)
end_time = time.time()

# Print the combined results
print("\nTOTAL RESULT")
for pmid in tot_pmids:
    print(pmid)
print(f"Time {end_time - start_time}")'''
