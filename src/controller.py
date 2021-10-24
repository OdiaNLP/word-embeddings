import os
from datetime import datetime
from time import time

from flask import Flask, render_template, request
from gensim.models import KeyedVectors

from src.form_model import InputFormWord2Vec
from src.common.log import WE_LOGGER
from src.utils import plot_embs, plot_dummy

# create app
app = Flask(__name__)

# set url postfix
rule = "/word2vec"

template_name = "my_view"

# create responses dir
os.makedirs("response", exist_ok=True)

# create plots dir
plots_dir = os.path.join("static")
os.makedirs(plots_dir, exist_ok=True)

# specify responses file path
responses_path = os.path.join("response", "word2vec_logs.txt")

if responses_path is not None:
    with open(responses_path, "a", encoding="utf-8") as f:
        WE_LOGGER.info(f"\nstarting app..[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}]\n")
        f.write(f"\nstarting app..[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}]\n")

model = KeyedVectors.load_word2vec_format(os.path.join("./src/model/embeddings.txt"))


@app.route(rule=rule, methods=["GET", "POST"])
def index():
    form = InputFormWord2Vec(request.form)
    if request.method == "POST" and form.validate():
        words = [word.strip() for word in form.words.data.split(",")]
        oov_words = [word for word in words if word not in model.wv.vocab]
        if len(oov_words) > 0:
            message = f'Word embeddings for [{", ".join(oov_words)}] does not exist. Choose different words..'
            WE_LOGGER.info(message)
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f"{str(time())}.png"))
        elif form.num_neighbours.data < 0:
            message = f"Set number of neighbours to a value >= 0."
            WE_LOGGER.info(message)
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f"{str(time())}.png"))
        else:
            message = ""
            image_path = plot_embs(
                model=model,
                base_words=words,
                num_neighbours=form.num_neighbours.data,
                title="Word embeddings",
                random_seed=123,
                save_path=os.path.join(plots_dir, f"{str(time())}.png"),
            )
        result = {"image_path": image_path, "message": message}
        WE_LOGGER.info("result:", result)
    else:
        result = None

    if result is not None and responses_path is not None:
        with open(responses_path, "a", encoding="utf-8") as fr:
            final_message = (f"\n\tNEW REQUEST 🤩 @"
                             f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                             f"\t[INPUTS] Base words: {form.words.data}, Number of neighbours: {form.num_neighbours.data}\n"
                             f"\t[OUTPUTS] Image path: {image_path}, Message: {message}\n")
            fr.write(final_message)
            WE_LOGGER(f"final_message: {final_message}")

    return render_template(template_name + ".html", form=form, result=result)
