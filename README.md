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
## Setup
### Train
- To train word embeddings, see the notebook [word2vec.ipynb](./docs/word2vec.ipynb).

### Web App Local Conda setup
- Clone the repo
- Put the `embeddings.txt` you have trained inside `src/model/` directory.
- Create a virtual environment. If installed Anaconda, you can try:
    ```shell
    $ conda create -n word_embeddings python=3.6
    ```
- Yes, we need Python 3.6 version for this.
- Install all the python dependencies with the following command:
    ```shell
    $ pip install -r requirements.txt
    ```
- Run the following command to run the server:
    ```shell
    $ gunicorn app:app -b 0.0.0.0:31137
    ```
- Now you can see the web app running in your browser at http://127.0.0.1:31137/word2vec
- If faced any error like below, please setup an environment variable `PYTHONIOENCODING` with value `utf-8`
    ```shell
    UnicodeEncodeError: 'ascii' codec can't encode character '\u2771' in position 1659: ordinal not in range(128)
    *** You may need to add PYTHONIOENCODING=utf-8 to your environment ***
    ```

### Web App Docker setup

1. Install Docker Desktop for [Mac](https://docs.docker.com/desktop/mac/install/) and [Windows](https://docs.docker.com/desktop/windows/install/) in your system.
2. Run the Docker. Type `docker` in your command prompt/terminal to check if this command is working.
3. Go to the project root folder i.e. `word-embeddings`.
4. Use the following command to build the image from Dockerfile

    ```shell
    docker build -t word_embeddings:latest .
    ```
5. Then you can run the following command to run the docker image.

    ```shell
    docker run --rm -it  -p 31137:31137 word_embeddings:latest
    ```

## Snapshot of web app
<img src="/docs/color_capture.png" width="75%" height="75%"/>

[LICENSE](https://github.com/OdiaNLP/word-embeddings/blob/main/LICENSE)
