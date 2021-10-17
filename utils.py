import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from matplotlib import font_manager as fm
from sklearn.manifold import TSNE


def plot_embs(model: KeyedVectors, base_words: List[str], num_neighbours: int, title: str,
              random_seed: int, save_path: str) -> str:
    """Plot embeddings for a set of words and their neighbours.
    First obtain the neighbours. Then apply tSNE. Finally plot."""

    embs = []
    all_viz_words = []
    is_base_word = []
    for word in base_words:
        embs.append(model.wv.get_vector(word))
        all_viz_words.append(word)
        is_base_word.append(True)
        for neighbour, _ in model.wv.most_similar(word, topn=num_neighbours):
            embs.append(model.wv.get_vector(neighbour))
            all_viz_words.append(neighbour)
            is_base_word.append(False)
    embs = np.stack(embs)

    # tSNE
    tsne_tokens_embedded = TSNE(n_components=2, metric='cosine', verbose=True,
                                random_state=random_seed).fit_transform(embs)
    x_tsne, y_tsne = zip(*tsne_tokens_embedded)

    # display scatter plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    for k, (label, is_base, x, y) in enumerate(zip(all_viz_words, is_base_word, x_tsne, y_tsne)):
        alpha = 1.0 if is_base else 0.2
        ax.scatter([x], [y], color='xkcd:blue', alpha=alpha)
        ax.text(x, y, label, color='xkcd:orangered', va='bottom', fontsize=12,
                fontproperties=get_odia_prop(os.path.join('fonts/OR51_Ananta.ttf')))
    ax.set_title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path)
    return save_path


def plot_dummy(save_path: str) -> str:
    """Create dummy plot"""
    _, _ = plt.subplots(figsize=(10, 8), dpi=160)
    plt.savefig(save_path)
    return save_path


def get_odia_prop(font_path: str):
    """Get Odia font properties from file"""
    return fm.FontProperties(fname=font_path)


if __name__ == "__main__":
    _model = KeyedVectors.load_word2vec_format(os.path.join('embeddings.txt'))
    _ = plot_embs(
        model=_model,
        base_words=['ଘୋଡା', 'ହାତୀ', 'ବାଘ', 'ଟିଭି', 'କଲମ', 'କାନ୍ଥ'],
        num_neighbours=5,
        title='Word embeddings',
        random_seed=123,
        save_path=os.path.join('example.png'),
    )
