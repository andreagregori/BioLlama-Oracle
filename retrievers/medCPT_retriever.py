import faiss
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel

# building the Faiss index of PubMed articles, let's use the flat inner product index
chunk_number = 37
pubmed_embeds = np.load(f"../../PubMed_embeddings/embeds_chunk_{chunk_number}.npy")
index = faiss.IndexFlatIP(768)
index.add(pubmed_embeds)

# these are the corresponding pmids for the article embeddings
pmids = json.load(open(f"../../PubMed_embeddings/pmids_chunk_{chunk_number}.json"))

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

queries = [
    "Is anaphylaxis a results of mast cell activation?"
]

with torch.no_grad():
    # tokenize the queries
    encoded = tokenizer(
        queries,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=64,
    )

    # encode the queries (use the [CLS] last hidden states as the representations)
    embeds = model(**encoded).last_hidden_state[:, 0, :]

    # search the Faiss index
    scores, inds = index.search(embeds, k=10)

# print the search results
for idx, query in enumerate(queries):
    print(f"Query: {query}")

    for score, ind in zip(scores[idx], inds[idx]):
        print(f"PMID: {pmids[ind]}; Score: {score}")
