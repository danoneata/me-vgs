import pdb
import random

from itertools import groupby
from toolz import first
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from matplotlib import pyplot as plt
from mevgs.data import AudioFeaturesLoader, MEDataset, Language, load_dictionary
from mevgs.utils import cache_np, cache_json, read_json
from mevgs.scripts.prepare_predictions_for_yevgen import load_model_and_config


sns.set_theme(context="talk", style="whitegrid", font="Arial", palette="tab10")

LANGS_LONG = ["english", "french", "dutch"]
LANGS_SHORT = ["en", "fr", "nl"]
LANG_SHORT_TO_LONG = {
    "en": "english",
    "fr": "french",
    "nl": "dutch",
}

DEFAULT_MODEL_PARAMS = {
    1: {
        "train-langs": ["en"],
        "model-size": 0,
        "has-links": 0,
    },
    2: {
        "train-langs": ["en", "fr"],
        "model-size": 0,
        "has-links": 0,
    },
}


def create_menu(id_):
    st.markdown(f"## Model {id_}")
    train_langs = st.multiselect(
        "Train languages:",
        LANGS_SHORT,
        DEFAULT_MODEL_PARAMS[id_]["train-langs"],
        key=f"train-langs-{id_}",
    )
    model_size = st.selectbox(
        "Model size:",
        ["sm", "md", "lg"],
        index=DEFAULT_MODEL_PARAMS[id_]["model-size"],
        key=f"model-size-{id_}",
    )
    has_links = st.selectbox(
        "Has links?",
        ["no", "yes"],
        index=DEFAULT_MODEL_PARAMS[id_]["has-links"],
        key=f"has-links-{id_}",
    )
    st.markdown("---")
    return {
        "id": id_,
        "train-langs": sorted(train_langs),
        "model-size": model_size,
        "has-links": has_links,
    }


def load_predictions(model_params, test_lang):
    filename = "{}_links-{}_size-{}_on-{}".format(
        "-".join(model_params["train-langs"]),
        model_params["has-links"],
        model_params["model-size"],
        test_lang,
    )
    df = pd.read_csv(f"output/preds/{filename}.csv")
    # df has columns "audio", "image-pos", "image-neg", "is-correct/0", "is-correct/1", ...
    # convert to multilevel columns
    df = df.set_index(["audio", "image-pos", "image-neg"], append=True)
    df.columns = pd.MultiIndex.from_tuples([col.split("/") for col in df.columns])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "i", "level_4": "seed"})
    df["model"] = model_params["id"]
    return df


def show_results(col, preds, title):
    scores = 100 * preds.groupby("seed")["is-correct"].mean()
    col.markdown(f"## {title}")
    col.markdown("Aggregated accuracy")
    col.markdown("{:.2f}±{:.1f}".format(scores.mean(), scores.std()))
    # col.markdown("Per seed performance")
    # col.dataframe(scores)
    # col.markdown("Per concept performance")
    # col.write(preds.groupby("audio")["is-correct"].mean())


def compare_results(model_params_1, model_params_2, test_lang):
    preds1 = load_predictions(model_params_1, test_lang)
    preds2 = load_predictions(model_params_2, test_lang)

    col1, col2 = st.columns(2)
    show_results(col1, preds1, "Model 1")
    show_results(col2, preds2, "Model 2")

    preds = pd.concat([preds1, preds2])

    # col1, col2 = st.columns(2)

    scores_per_seed = 100 * preds.groupby(["model", "seed"])["is-correct"].mean()
    scores_per_seed = scores_per_seed.reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=scores_per_seed, x="is-correct", y="seed", hue="model", ax=ax)
    sns.move_legend(ax, "upper left")

    st.markdown("## Per seed accuracy")
    st.pyplot(fig, use_container_width=False)

    scores_per_concept = (
        100 * preds.groupby(["model", "seed", "audio"])["is-correct"].mean()
    )
    scores_per_concept = scores_per_concept.reset_index()
    fig, ax = plt.subplots(figsize=(5, 12))
    sns.stripplot(
        data=scores_per_concept,
        x="is-correct",
        y="audio",
        hue="model",
        # dodge=0.1,
        # dodge=True,
        # alpha=0.5,
        # legend=False,
        ax=ax,
    )
    # sns.pointplot(
    # data=scores_per_concept,
    # x="is-correct",
    # y="audio",
    # hue="model",
    # dodge=True,
    # linestyle="none",
    # errorbar=None,
    # marker="d",
    # markersize=5,
    # ax=ax,
    # )
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_xlim(0, 100)
    ax.axvline(50, color="red", alpha=0.2, linestyle="--")
    sns.move_legend(ax, "upper left")
    # fig.set_tight_layout()

    st.markdown("## Per concept accuracy")
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")


def compute_audio_embeddings(model_name, test_lang, audio_data, seed):
    test_lang_1 = LANG_SHORT_TO_LONG[test_lang]
    model, config = load_model_and_config(model_name, test_lang_1, seed)
    device = config["device"]

    audio_features_loader = AudioFeaturesLoader("wavlm-base-plus", "test", LANGS_LONG)

    def compute_audio_embedding(datum):
        audio = audio_features_loader(datum)
        audio = audio.unsqueeze(0)
        audio = audio.to(device)
        audio_length = torch.tensor([audio.size(2)])
        audio_length = audio_length.to(device)
        return model.audio_enc(audio, audio_length)

    with torch.no_grad():
        embs = [compute_audio_embedding(datum) for datum in tqdm(audio_data)]

    embs = torch.cat(embs)
    embs = embs.squeeze(2)
    embs = model.l2_normalize(embs, 1)
    return embs.cpu().numpy()


def show_audio_similarities(model_params, test_lang):
    random.seed(1337)

    def get_sample(group, n):
        group = list(group)
        if len(group) <= n:
            return group
        else:
            return random.sample(list(group), n)

    def load_audio_data_subset(dataset, n):
        audio_files = dataset.audio_files
        key = lambda datum: (datum["word-en"], datum["lang"])
        audio_files = sorted(audio_files, key=key)
        return [
            datum
            for _, group in groupby(audio_files, key)
            for datum in get_sample(group, n)
        ]

    n = 30
    dataset = MEDataset("test", LANGS_LONG)  # type: ignore
    path = f"output/show-model-comparison/audio-data-ss-{n}.json"
    data = cache_json(path, load_audio_data_subset, dataset, n=n)

    model_name = "{}_links-{}_size-{}".format(
        "-".join(model_params["train-langs"]),
        model_params["has-links"],
        model_params["model-size"],
    )
    embs = cache_np(
        f"output/show-model-comparison/embeddings/{model_name}_seed-a.npy",
        compute_audio_embeddings,
        model_name=model_name,
        test_lang=test_lang,
        audio_data=data,
        seed="a",
    )

    def find_indices(w):
        return [i for i, datum in enumerate(data) if datum["lang"] == LANG_SHORT_TO_LONG[w["lang"]] and datum["word-en"] == w["word-en"]]

    def compute_similarity(w1, w2, emb):
        idxs1 = find_indices(w1)
        idxs2 = find_indices(w2)
        sims = emb[idxs1] @ emb[idxs2].T
        return np.mean(sims)

    words_tr = [
        {
            "lang": lang,
            "word-en": word,
        }
        for lang in model_params["train-langs"]
        for word in dataset.words_seen
    ]
    words_te = [
        {
            "lang": test_lang,
            "word-en": word,
        }
        for word in dataset.words_unseen
    ]

    words = load_dictionary()

    def get_word_str(w):
        if w["lang"] == "en":
            return w["word-en"]
        else:
            word = w["word-en"]
            lang = w["lang"]
            lang_long = LANG_SHORT_TO_LONG[lang]
            word_trans = first(entry[lang_long] for entry in words if entry["english"] == word)
            return "{} ({})".format(word, word_trans)

    similarities = [
        {
            "word tr": get_word_str(w1),
            "word te": get_word_str(w2),
            "similarity": compute_similarity(w1, w2, embs),
        }
        for w1 in words_tr
        for w2 in words_te
    ]
    df = pd.DataFrame(similarities)

    df = df.pivot(index="word tr", columns="word te", values="similarity")
    S = 0.8
    h = len(words_tr) * S
    w = len(words_te) * S
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        linewidth=0.5,
        # cmap="bwr",
        square=True,
        cbar=False,
        ax=ax,
    )
    fig.tight_layout()
    return fig


def compare_embeddings(model_params_1, model_params_2, test_lang):
    fig1 = show_audio_similarities(model_params_1, test_lang)
    fig2 = show_audio_similarities(model_params_2, test_lang)
    # cols = st.columns(2)
    st.markdown("Model 1")
    st.pyplot(fig1)
    st.markdown("---")

    st.markdown("Model 2")
    st.pyplot(fig2)
    st.markdown("---")


def main():
    # st.set_page_config(layout="wide")
    with st.sidebar:
        model_params_1 = create_menu(1)
        model_params_2 = create_menu(2)

        train_langs_1 = set(model_params_1["train-langs"])
        train_langs_2 = set(model_params_2["train-langs"])

        test_langs = train_langs_1 & train_langs_2
        test_lang = st.selectbox("Test language:", list(test_langs))

    if not test_langs:
        st.error("No common test language.")
    else:
        st.markdown(
            """
        Below you can see a comparison of the mutual exclusivity bias (novel–familiar accuracy) of the two selected models.
        The results are presented three levels of granularity: aggregated, per seed, per concept and seed.
        """
        )
        compare_results(model_params_1, model_params_2, test_lang)
        compare_embeddings(model_params_1, model_params_2, test_lang)


if __name__ == "__main__":
    main()
