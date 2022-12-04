import lightgbm as lgb
import numpy as np
import random

from tqdm import tqdm
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

class Documents:
    def __init__(self, path:str):
        self.documents = {}
        with open(path, encoding='UTF-8') as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()

class Queries:
    def __init__(self, path:str):
        self.queries = {}
        with open(path, encoding='UTF-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        
class Letor:
    def __init__(self, doc_path, qry_path, qrel_path):
        self.NUM_NEGATIVES = 1
        self.NUM_LATENT_TOPICS = 200

        self.documents = Documents(doc_path)
        self.queries = Queries(qry_path)
        self.q_docs_rel = {}
        self.init_qrel(qrel_path)
        self.genr_dataset()
        self.lsa_model()
        self.data_set()
        self.rank_model()


    # ============================================= TRAINING DATASET ============================================= #
    def init_qrel(self, qrel_path):
        with open(qrel_path) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries.queries) and (doc_id in self.documents.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

    def genr_dataset(self):
        self.group_qid_count = []
        self.dataset = []
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries.queries[q_id], self.documents.documents[doc_id], rel))
        
            self.dataset.append((self.queries.queries[q_id], random.choice(list(self.documents.documents.values())), 0))

        self.bow_corpus = [Dictionary().doc2bow(doc, allow_update = True) for doc in self.documents.documents.values()]


    # ============================================= BUILDING LSA MODEL ============================================= #
    def lsa_model(self):
        self.model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS)

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[Dictionary().doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def data_set(self):
        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X,Y


    # ============================================= TRAINING THE RANKER ============================================= #
    def rank_model(self):
        X, Y = self.data_set()
        self.ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)

        self.ranker.fit(X, Y,
                group = self.group_qid_count,
                verbose = 10)

    def predict(self, query, docs):
        X_unseen = []
        for doc_id, doc in tqdm(docs):
            X_unseen.append(self.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        for (did, score) in sorted_did_scores:
            print(did, score)
