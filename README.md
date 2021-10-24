# Odia Word Embeddings

Train Odia word embeddings using word2vec.

## Dependencies
See the dependencies in `requirements.txt`.
The code has been tested with Python 3.6.

## Overview

Check out [this blog post](https://jalammar.github.io/illustrated-word2vec/) to get an illustrated guide 📙 to word2vec.

- First download Odia text data.

```shell
mkdir data
cd data
!wget https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/sentence/or.txt.gz
tar -zxvf or.txt.gz
head or
```

- To train word embeddings, see the notebook `word2vec.ipynb`.
- Finally run `controller.py` to start the web app. Go to http://127.0.0.1:31137/word2vec to access the web app.

```shell
# web app
python controller.py  # open http://127.0.0.1:31137/word2vec in browser
```

## Snapshot of web app
<img src="/docs/snapshot.png" width="50%" height="50%"/>

[LICENSE](https://github.com/OdiaNLP/word-embeddings/blob/main/LICENSE)
