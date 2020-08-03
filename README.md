# Transformer-as-a-Service

Using BERT model as a sentence encoding service, i.e. mapping a variable-length sentence to a fixed-length vector.


* ``server.py:`` Transformer server; fastAPI application to encode the input and return embeddings

* ``client.py`` Transformer client; Abstraction to connect with transformer server and send/receive query/embeddings

* Model used: [bert-base-nli-mean-tokens](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/bert-base-nli-mean-tokens.zip) ([sentence-transformers](https://pypi.org/project/sentence-transformers/))


#### Client usage


```python
t = TransformerClient()  # specify host and port if necessary
t.encode("What happens if I get an NQ")
```