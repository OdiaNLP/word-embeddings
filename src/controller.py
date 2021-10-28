import os
from datetime import datetime
from time import time

from flask import render_template, request

from src.common.log import WE_LOGGER
from src.form_model import InputFormWord2Vec
from src.utils import plot_embs, plot_dummy


def process_word2vec(plots_dir, responses_path, model):
    form = InputFormWord2Vec(request.form)
    if request.method == "POST" and form.validate():
        words = [word.strip() for word in form.words.data.split(",")]
        WE_LOGGER.info(f"Words received: {words}")
        oov_words = [word for word in words if word not in model.wv.vocab]
        WE_LOGGER.info(f"OOV Words received: {oov_words}")
        if len(oov_words) > 0:
            message = f'Word embeddings for [{", ".join(oov_words)}] does not exist. Choose different words..'
            WE_LOGGER.info(message)
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f"{str(time())}.png"))
        elif form.num_neighbours.data < 0:
            message = f"Set number of neighbours to a value >= 0."
            WE_LOGGER.info(message)
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f"{str(time())}.png"))
        else:
            WE_LOGGER.info("Now in the else clause.")
            message = ""
            image_path = plot_embs(
                model=model,
                base_words=words,
                num_neighbours=form.num_neighbours.data,
                title="Word embeddings",
                random_seed=123,
                save_path=os.path.join(plots_dir, f"{str(time())}.png"),
            )
            WE_LOGGER.info(f"Image path: {image_path}")
        result = {"image_path": image_path, "message": message}
        WE_LOGGER.info(f"result: {result}")
    else:
        result = None

    if result is not None and responses_path is not None:
        with open(responses_path, mode="a", encoding="utf-8") as fr:
            final_message = (
                f"\n\tNEW REQUEST 🤩 @"
                f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                f"\t[INPUTS] Base words: {form.words.data}, Number of neighbours: {form.num_neighbours.data}\n"
                f"\t[OUTPUTS] Image path: {image_path}, Message: {message}\n"
            )
            fr.write(final_message)
            # WE_LOGGER(f"final_message: {final_message}")
    return render_template("my_view.html", form=form, result=result)
