import os
from datetime import datetime

from flask import Flask
from gensim.models import KeyedVectors

from src.common.log import WE_LOGGER
from src.controller import process_word2vec

app = Flask(__name__)
os.makedirs("response", exist_ok=True)
responses_path = os.path.join("response", "word2vec_logs.txt")
plots_dir = os.path.join("static")
os.makedirs(plots_dir, exist_ok=True)

if responses_path is not None:
    with open(responses_path, "a", encoding="utf-8") as f:
        WE_LOGGER.info(f"\nstarting app..[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}]\n")
        f.write(f"\nstarting app..[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}]\n")

model = KeyedVectors.load_word2vec_format(os.path.join("./src/model/embeddings.txt"))


@app.route("/", methods=["GET"])
def index():
    return "It is working!"


@app.route("/word2vec", methods=["GET", "POST"])
def word2vec():
    return process_word2vec(plots_dir, responses_path, model)
