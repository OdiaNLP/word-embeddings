import os
from datetime import datetime
from time import time

from flask import Flask, render_template, request
from gensim.models import KeyedVectors

from form_model import InputFormWord2Vec
from utils import plot_embs, plot_dummy

# create app
app = Flask(__name__)

# set url postfix
rule = '/word2vec'


@app.route(rule=rule, methods=['GET', 'POST'])
def index():
    form = InputFormWord2Vec(request.form)
    if request.method == 'POST' and form.validate():
        words = [word.strip() for word in form.words.data.split(',')]
        oov_words = [word for word in words if word not in model.wv.vocab]
        if len(oov_words) > 0:
            message = f'Word embeddings for [{", ".join(oov_words)}] does not exist. Choose different words..'
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f'{str(time())}.png'))
        elif form.num_neighbours.data < 0:
            message = f'Set number of neighbours to a value >= 0.'
            image_path = plot_dummy(save_path=os.path.join(plots_dir, f'{str(time())}.png'))
        else:
            message = ''
            image_path = plot_embs(model=model, base_words=words, num_neighbours=form.num_neighbours.data,
                                   title='Word embeddings', random_seed=123,
                                   save_path=os.path.join(plots_dir, f'{str(time())}.png'))
        result = {'image_path': image_path, 'message': message}
    else:
        result = None

    if result is not None and responses_path is not None:
        with open(responses_path, 'a', encoding='utf-8') as fr:
            fr.write(
                f'\n\tNEW REQUEST 🤩 @'
                f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                f'\t[INPUTS] Base words: {form.words.data}, Number of neighbours: {form.num_neighbours.data}\n'
                f'\t[OUTPUTS] Image path: {image_path}, Message: {message}\n'
            )

    return render_template(template_name + '.html', form=form, result=result)


if __name__ == '__main__':

    # set template name
    template_name = 'my_view'

    # create responses dir
    os.makedirs('responses', exist_ok=True)

    # create plots dir
    plots_dir = os.path.join('static')
    os.makedirs(plots_dir, exist_ok=True)

    # specify responses file path
    responses_path = os.path.join('responses', f'{rule[1:]}_logs.txt')

    if responses_path is not None:
        with open(responses_path, 'a', encoding='utf-8') as f:
            f.write(
                f'\nstarting app.. '
                f'[{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'
                f'\n'
            )

    # load word vectors
    model = KeyedVectors.load_word2vec_format(os.path.join('embeddings.txt'))

    # run app
    app.run(host='127.0.0.1', port=31137, debug=False)
