import pdb
import json
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestCentroid

from mevgs.data import MEDataset
from mevgs.utils import cache_np
from mevgs.scripts.show_intra_audio_sim_lang_links import extract_embeddings

LANGS = ("english", "french")


def evaluate_model(model_spec):

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

    def eval1(lang_src, lang_tgt):
        X_tr, y_tr = get_X_y(lang_src)
        X_te, y_te = get_X_y(lang_tgt)
        ncm = NearestCentroid()
        return ncm.fit(X_tr, y_tr).score(X_te, y_te)
        # ncm.fit(X_tr, y_tr)
        # class_to_centroid = {cls: centroid for cls, centroid in zip(ncm.classes_, ncm.centroids_)}
        # C_te = [class_to_centroid[cls] for cls in y_te]
        # C_te = np.array(C_te)
        # return np.einsum("ij,ij->i", X_te, C_te).mean()

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
    results = [
        evaluate_model(f"en-fr{l}-{v}")
        for l in ("", "-links")
        for v in "abcde"
    ]
    df = pd.DataFrame(results)
    # df = df.stack()
    print(df.to_csv(index=False))
