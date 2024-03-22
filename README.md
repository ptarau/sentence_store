# sentence_store
### A tool to extract and store sentence embeddings to a fast and scalable vector db

#### Install:

pip3 install sentence_store

#### Usage:

```
from sentence_store.main import Embedder
 e = Embedder(cache_name='embedder_test')
    sents = [
        "The dog barks to the moon",
        "The cat sits on the mat",
        "The phone rings",
        "The rocket explodes",
        "The cat and the dog sleep"
    ]
    e.store(sents)
    q = 'Who sleeps on the mat?'
    rs = e(q, 2)
    for r in rs: print(r)

    print('TIMES:', e.times)
    return True
```

Enjoy,
Paul Tarau
