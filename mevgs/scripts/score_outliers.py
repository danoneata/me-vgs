from typing import List

import streamlit as st
import torch

from sklearn.svm import OneClassSVM

from mevgs.data import MEDataset, Split, AudioFeaturesLoader, get_audio_path
from mevgs.utils import read_file, cache_json


SPLITS = ["train", "valid", "test"]  # type: List[Split]


def score_outliers(feat_type, lang, word):
    datasets = {split: MEDataset(split, (lang,)) for split in SPLITS}
    load_features = {
        split: AudioFeaturesLoader(feat_type, split, (lang,)) for split in SPLITS
    }
    features = [
        load_features[split](datum).mean(dim=1)
        for split, dataset in datasets.items()
        for datum in dataset.word_to_audios.get(word, [])
    ]
    features = torch.stack(features, dim=0).numpy()
    model = OneClassSVM(kernel="rbf", nu=0.1)
    model.fit(features)
    scores = model.decision_function(features)
    data = []
    i = 0
    for split, dataset in datasets.items():
        for datum in dataset.word_to_audios.get(word, []):
            data.append(
                {
                    **datum,
                    "score": scores[i],
                    "split": split,
                }
            )
            i += 1
    return data


def show_results():
    words_seen = read_file("data/words-seen.txt")
    words_unseen = read_file("data/words-unseen.txt")
    words = words_seen + words_unseen

    with st.sidebar:
        feat_type = st.selectbox("Feature type", ["wavlm-base-plus", "wav2vec2-xls-r-2b"], index=0)
        word = st.selectbox("Word", words)

    data = cache_json(f"output/score-outliers/{feat_type}-{word}.json", score_outliers, feat_type, "dutch", word)
    data = sorted(data, key=lambda x: x["score"])

    def show_results_col(col, data, title):
        col.markdown("## {}".format(title))
        for datum in data:
            col.markdown("`{}` · score: {:.1f}".format(datum["name"], datum["score"]))
            path = get_audio_path(datum)
            col.audio(path)

    col1, col2 = st.columns(2)
    show_results_col(col1, data[:10], "Bottom 10 (outliers)")
    show_results_col(col2, data[-10:], "Top 10 (inliers)")


show_results()