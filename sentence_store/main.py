from time import time
from collections import Counter

from sentence_store.tools import (
    to_json, from_json, exists_file
)
import torch
from sentence_transformers import SentenceTransformer
from vecstore.vecstore import VecStore


# SBERT API

def sbert_embed(sents, emebedding_model="all-MiniLM-L6-v2"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(emebedding_model, device=device)
    embeddings = model.encode(sents, show_progress_bar=True)
    return embeddings


class Embedder:
    """
    embeds a set of sentences using an LLM
    and store them into a vector store
    """

    def __init__(self, cache_name):
        assert cache_name is not None
        self.total_toks = 0
        self.cache_name = cache_name
        self.CACHES = "./SENT_STORE_CACHE/"
        self.emebedding_model = None
        self.vstore = None
        self.times = Counter()

    def cache(self, ending):
        return self.CACHES + self.cache_name + ending

    def embed(self, sents):
        t1 = time()
        embeddings = sbert_embed(sents)
        t2 = time()
        self.times['embed'] += t2 - t1
        return embeddings

    def clear(self):
        fj = self.cache('.json')
        fb = self.cache('.bin')


    def store(self, sents):
        """
        embeds and caches the sentences and their embeddings
        unleass this is already done or force=True
        """
        fj = self.cache('.json')
        fb = self.cache('.bin')
        if exists_file(fj) and exists_file(fb):
            self.load()
            return

        embeddings = self.embed(sents)
        dim = embeddings.shape[1]
        if self.vstore is None:
            self.vstore = VecStore(fb, dim=dim)
        self.vstore.add(embeddings)

        to_json((dim, sents), fj)
        self.vstore.save()

    def load(self):
        fj = self.cache('.json')
        fb = self.cache(ending='.bin')
        dim, sents = from_json(fj)
        self.vstore = VecStore(fb, dim=dim)
        self.vstore.load()
        return sents

    def query(self, query_sent, top_k):
        """
        fetches the store
        """
        t1 = time()
        sents = self.load()
        query_embeddings = self.embed([query_sent])
        t2 = time()
        self.times['query'] += t2 - t1
        knn_pairs = self.vstore.query_one(query_embeddings[0], k=top_k)
        answers = [(sents[i], r) for (i, r) in knn_pairs]
        return answers

    def knns(self, top_k, as_weights=True):
        assert top_k > 0, top_k
        t1 = time()
        self.load()
        assert self.vstore is not None
        knn_pairs = self.vstore.all_knns(k=top_k, as_weights=as_weights)
        t2 = time()
        self.times['knns'] += t2 - t1
        return knn_pairs

    def get_sents(self):
        return from_json(self.cache('.json'))[1]

    def __call__(self, quest, top_k):
        return self.query(quest, top_k)

    def get_times(self):
        return self.times | self.vstore.times


def test_main():
    e = Embedder(cache_name='embedder_test')
    sents = [
        "The dog barks to the moon",
        "The cat sits on the mat",
        "The phone rings",
        "The rocket explodes",
        "The cat and the dog sleep",
        "The cellphone is on the table"
    ]
    e.store(sents)
    q = 'Who sleeps on the mat?'
    rs = e(q, 2)
    for r in rs: print(r)

    print("\nCOMPUTING KNNS for k=3:")

    as_weights = False
    print('KNNS DONE:', as_weights)
    rs = e.knns(3, as_weights=as_weights)
    for i, r in enumerate(rs):
        print(i, sents[i])
        for x, v in r:
            print('    ', x, sents[x], v)
    print()

    as_weights = True
    print('KNNS DONE:', as_weights)
    rs = e.knns(3, as_weights=as_weights)
    for r in rs: print(r)
    print()

    print('TIMES:', e.times)
    return True


def test_big(url='https://www.gutenberg.org/cache/epub/2600/pg2600.txt'):
    import urllib.request
    with urllib.request.urlopen(url) as f:
        text = f.read().decode('utf-8')
        print('TEXT:', text[0:50], '...')
        sents = text.split('\n')
        sents = [s.strip() for s in sents if s.strip()]
        e = Embedder(cache_name='big_test')
        print('SENTS:', len(sents))
        print('COMPUTING AND STORING EMBEDDINGS')
        e.store(sents)
        print('DIMS:', e.vstore)
        print("COMPUTING KNNS for k=3:")
        print('DONE:', len(e.knns(3, as_weights=True)))
        print('QUERY WITH 3 ANSWERS:')
        rs = e('What did Napoleon say when he arrived to Moscow?', 3)
        print('RETRIEVED:\n')
        for r in rs: print(*r)
        print('\nTIMES:\n')
        for x in e.get_times().items(): print(x)
        return True


if __name__ == "__main__":
    # assert test_big()
    assert test_main()
