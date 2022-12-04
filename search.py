from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import numpy as np
import re
# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("Query  : ", query)
    print("Results:")
    model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
    doc_ids = []
    X_unseen = []
    docs = []

    print("===========Dengan TF-IDF==============")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        doc_ids.append(doc)
        print(f"{doc:30} {score:>.3f}")

    # print("===========Dengan BM25==============")
    # for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
    #     doc_ids.append(doc)
    #     print(f"{doc:30} {score:>.3f}")

    print("===========Dengan LETOR==============")
    for id in doc_ids:
        doc_text = ""
        with open(id) as file:
            for line in file:
                doc_text += line
            doc_text = re.sub(r"\s+", " ", doc_text)
        docs.append((id, doc_text))

    model.predict(query, docs)

    print()
