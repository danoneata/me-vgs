import pdb
import sys

import json
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestCentroid

from mevgs.data import MEDataset
from mevgs.utils import cache_np
from mevgs.scripts.show_intra_audio_sim_lang_links import extract_embeddings

LANGS = ("english", "french")


def evaluate_model(model_spec, score_type):

    def load_audios(lang):
        path = f"output/scripts-cache/selected-audios-{lang}.json"
        with open(path) as f:
            return json.load(f)

    datasets = {lang: MEDataset("test", (lang,)) for lang in LANGS}
    audios = {lang: load_audios(lang) for lang in LANGS}

    def get_embs(lang):
        path = f"output/scripts-cache/features-{model_spec}-{lang}.npy"
        return cache_np(
            path,
            extract_embeddings,
            datasets[lang],
            audios[lang],
            model_spec,
        )

    embs = {lang: get_embs(lang) for lang in LANGS}

    def get_X_y(lang):
        idxs = [
            i
            for i, audio in enumerate(audios[lang])
            if audio["word-en"] in datasets["english"].words_seen
        ]
        X = embs[lang][idxs]
        y = [
            audio["word-en"]
            for audio in audios[lang]
            if audio["word-en"] in datasets["english"].words_seen
        ]
        return X, y

    def l2_normalize(X):
        return X / np.linalg.norm(X, axis=1)[:, None]

    def eval1(lang_src, lang_tgt):
        X_tr, y_tr = get_X_y(lang_src)
        X_te, y_te = get_X_y(lang_tgt)
        ncm = NearestCentroid()
        ncm.fit(X_tr, y_tr)
        ncm.centroids_ = l2_normalize(ncm.centroids_)

        def get_accuracy():
            # import scipy.spatial as sp
            # sim = 1 - sp.distance.cdist(X_te, ncm.centroids_, "cosine")
            # y_pred = sim.argmax(axis=1)
            # y_pred = ncm.classes_[y_pred]
            # return (y_pred == y_te).mean()
            return ncm.score(X_te, y_te)

        def get_cos_sim():
            class_to_centroid = {cls: centroid for cls, centroid in zip(ncm.classes_, ncm.centroids_)}
            C_te = [class_to_centroid[cls] for cls in y_te]
            C_te = np.array(C_te)
            cos_sim = np.einsum("ij,ij->i", X_te, C_te)
            return cos_sim.mean()

        SCORE_FNS = {
            "accuracy": get_accuracy,
            "cos-sim": get_cos_sim,
        }
        return SCORE_FNS[score_type]()


    en_to_fr = 100 * eval1("english", "french")
    fr_to_en = 100 * eval1("french", "english")
    mean = (en_to_fr + fr_to_en) / 2

    return {
        "model": model_spec,
        "en → fr": en_to_fr,
        "fr → en": fr_to_en,
        "mean": mean,
    }


if __name__ == "__main__":
    eval_type = sys.argv[1]
    results = [
        evaluate_model(f"en-fr{l}-{v}", eval_type)
        for l in ("", "-links")
        for v in "abcde"
    ]
    df = pd.DataFrame(results)
    # df = df.stack()
    print(df.to_csv(index=False))
