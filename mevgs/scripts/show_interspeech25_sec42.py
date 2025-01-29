import pdb

from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA

from adjustText import adjust_text

from mevgs.data import AudioFeaturesLoader
from mevgs.utils import read_json, cache_json

from mevgs.scripts.show_explain_me import (
    load_data_and_embs_all,
    WORDS_SEEN,
    WORDS_UNSEEN,
    LANG_SHORT_TO_LONG,
)
from mevgs.scripts.show_crosslingual_audio_alignment import (
    add_texts,
    align,
    translate,
)

sns.set(context="poster", style="white", font="Arial")

LANG_LONG_TO_SHORT = {v: k for k, v in LANG_SHORT_TO_LONG.items()}

WORDS = {
    "seen": set(WORDS_SEEN),
    "unseen": set(WORDS_UNSEEN),
}


def filter(data, embs, to_keep):
    idxs = [i for i, datum in enumerate(data) if to_keep(datum)]
    data1 = [data[i] for i in idxs]
    embs1 = embs[idxs]
    return data1, embs1


def clear_axes(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def show_2d(data, embs_2d, modality, langs, ax):
    def get_title(langs):
        prefix = "Monolingual" if len(langs) == 1 else "Bilingual"
        langs = [LANG_SHORT_TO_LONG[lang].capitalize() for lang in langs]
        return "{}: {}".format(prefix, ", ".join(langs))

    df = pd.DataFrame(embs_2d, columns=["x", "y"])
    df["word-en"] = [d["word-en"] for d in data]

    if modality == "audio":
        df["lang"] = [d["lang"] for d in data]
        df["word"] = [translate(d["word-en"], d["lang"]) for d in data]
        kwargs = {
            "style": "lang",
            "markers": {
                "english": "o",
                "french": "X",
                "dutch": "s",
            },
        }
    else:
        kwargs = {
            "marker": "P",
        }

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="word-en",
        legend=False,
        ax=ax,
        **kwargs,
    )

    if modality == "audio":
        add_texts(ax, df, fontsize=13)

    ax.set_title(get_title(langs))

    return ax


def do1(train_langs, size, seed):
    modalities = ["audio", "image"]
    langs_combinations = [
        (train_langs[0],),
        (train_langs[1],),
        train_langs,
    ]

    words = set(WORDS_SEEN)

    def to_keep(datum, modality, langs):
        cond1 = datum["word-en"] in words
        cond2 = modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs
        return cond1 and cond2

    data_embs = {
        (modality, langs): filter(
            *load_data_and_embs_all(
                modality,
                langs,
                size,
                seed,
            ),
            partial(to_keep, modality=modality, langs=langs),
        )
        for modality in modalities
        for langs in langs_combinations
    }

    anchor = ("audio", train_langs)

    def align1(k):
        data1, embs1 = data_embs[k]
        data2, embs2 = data_embs[anchor]
        embs, error = align(embs1, embs2, data1, data2, None)
        print(k, error)
        return embs

    embs_aligned = {k: align1(k) for k in data_embs}

    anchor_embs = data_embs[anchor][1]
    proj = PCA(n_components=2)
    proj.fit(anchor_embs)

    # def project2d(embs):
    #     proj = PCA(n_components=2)
    #     return proj.fit_transform(embs)
    #     # proj = TSNE(n_components=2, random_state=0)
    #     # return proj.fit_transform(embs)

    # embs = {k: project2d(v[1]) for k, v in data_embs.items()}

    # def align2(k):
    #     data1, _ = data_embs[k]
    #     data2, _ = data_embs[anchor]
    #     embs1, error = align(embs[k], embs[anchor], data1, data2, None)
    #     st.write(k, error)
    #     return embs1

    # embs_aligned = {k: align2(k) for k in data_embs}
    # # embs_aligned = embs

    S = 5
    nrows = 2
    ncols = 3
    sns.set(context="poster", style="white", font="Arial")
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(S * ncols, S * nrows),
        sharex=True,
        sharey=True,
    )
    for r, modality in enumerate(modalities):
        for c, langs in enumerate(langs_combinations):
            data, _ = data_embs[(modality, langs)]
            embs = embs_aligned[(modality, langs)]

            ax = axs[r, c]
            # ax = show_2d(data, embs, modality, langs, ax)
            ax = show_2d(data, proj.transform(embs), modality, langs, ax)

            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if c == 0:
                ax.set_ylabel("{} embeddings".format(modality.capitalize()))

    fig.set_tight_layout(True)
    fig.savefig("output/interspeech25/embeddings-monolingual-vs-bilingual.pdf")
    st.pyplot(fig)


def do2(train_langs, size, seed):
    def to_keep(datum, modality, langs):
        return modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs

    data_audio, embs_audio = filter(
        *load_data_and_embs_all("audio", train_langs, size, seed),
        partial(to_keep, modality="audio", langs=train_langs),
    )
    data_image, embs_image = filter(
        *load_data_and_embs_all("image", train_langs, size, seed),
        partial(to_keep, modality="image", langs=train_langs),
    )

    for datum in data_audio:
        datum["modality"] = "audio"
        datum["is-seen"] = datum["word-en"] in WORDS_SEEN
        datum["type"] = datum["lang"]

    for datum in data_image:
        datum["modality"] = "image"
        datum["is-seen"] = datum["word-en"] in WORDS_SEEN
        datum["type"] = "image"

    data = data_audio + data_image
    embs = np.concatenate([embs_audio, embs_image], axis=0)

    proj = PCA(n_components=2, random_state=0)
    embs_2d = proj.fit_transform(embs)

    df = pd.DataFrame(data)
    df["x"] = embs_2d[:, 0]
    df["y"] = embs_2d[:, 1]

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    markers = {
        "english": "o",
        "french": "X",
        "dutch": "s",
        "image": "P",
    }

    # st.write(df)

    idxs = ~df["is-seen"]
    sns.scatterplot(
        data=df[idxs],
        x="x",
        y="y",
        color="gray",
        style="type",
        legend=False,
        alpha=0.5,
        markers=markers,
        ax=ax1,
    )

    idxs = df["is-seen"]
    sns.scatterplot(
        data=df[idxs],
        x="x",
        y="y",
        hue="word-en",
        style="type",
        legend=False,
        markers=markers,
        ax=ax1,
    )

    clear_axes(ax1)
    ax1.set_title("Audio and image embeddings")

    # centroids = df.groupby(["modality", "word-en"])["x", "y"].mean()
    # for word in WORDS_SEEN:
    #     audio_centroid = centroids.loc[("audio", word)]
    #     image_centroid = centroids.loc[("image", word)]
    #     ax.plot(
    #         [audio_centroid["x"], image_centroid["x"]],
    #         [audio_centroid["y"], image_centroid["y"]],
    #         "k--",
    #         c="gray",
    #         alpha=0.5,
    #         linewidth=1,
    #     )

    # sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    def show_2d(data, embs_2d, modality, langs, ax):
        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["word-en"] = [d["word-en"] for d in data]

        if modality == "audio":
            df["lang"] = [d["lang"] for d in data]
            df["word"] = [translate(d["word-en"], d["lang"]) for d in data]
            kwargs = {
                "style": "lang",
                "markers": {
                    "english": "o",
                    "french": "X",
                    "dutch": "s",
                },
            }
        else:
            kwargs = {
                "marker": "P",
            }

        idxs = df["word-en"].isin(WORDS_SEEN)

        sns.scatterplot(
            data=df[~idxs],
            x="x",
            y="y",
            color="gray",
            alpha=0.5,
            legend=False,
            ax=ax,
            **kwargs,
        )

        sns.scatterplot(
            data=df[idxs],
            x="x",
            y="y",
            hue="word-en",
            legend=False,
            ax=ax,
            **kwargs,
        )

        if modality == "audio":
            add_texts(ax, df[idxs], fontsize=16)
        else:
            df2 = df.copy()
            df2["word"] = df2["word-en"]
            add_texts(ax, df2[idxs], fontsize=16, key="word")

        ax.set_title("{} embeddings".format(modality.capitalize()))
        clear_axes(ax)

    proj = TSNE(n_components=2, random_state=0)
    # proj = PCA(n_components=2, random_state=0)

    show_2d(data_audio, proj.fit_transform(embs_audio), "audio", train_langs, ax2)
    show_2d(data_image, proj.fit_transform(embs_image), "image", train_langs, ax3)

    fig.set_tight_layout(True)
    st.pyplot(fig)
    fig.savefig("output/interspeech25/embeddings-audio-vs-image.pdf")


def do3(train_langs, size, seed):
    def to_keep(datum, modality, langs):
        return modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs

    data_audio, embs_audio = filter(
        *load_data_and_embs_all("audio", train_langs, size, seed),
        partial(to_keep, modality="audio", langs=train_langs),
    )
    data_image, embs_image = filter(
        *load_data_and_embs_all("image", train_langs, size, seed),
        partial(to_keep, modality="image", langs=train_langs),
    )

    for datum in data_audio:
        datum["modality"] = "audio"
        datum["is-seen"] = datum["word-en"] in WORDS_SEEN
        datum["type"] = datum["lang"]

    for datum in data_image:
        datum["modality"] = "image"
        datum["is-seen"] = datum["word-en"] in WORDS_SEEN
        datum["type"] = "image"

    data = data_audio + data_image
    embs = np.concatenate([embs_audio, embs_image], axis=0)

    proj = PCA(n_components=2, random_state=0)
    embs_2d = proj.fit_transform(embs)

    df = pd.DataFrame(data)
    df["x"] = embs_2d[:, 0]
    df["y"] = embs_2d[:, 1]

    fig, axs = plt.subplots(figsize=(11, 6.5), ncols=2, nrows=1)

    markers = {
        "english": "o",
        "french": "X",
        "dutch": "s",
        "image": "P",
    }

    # st.write(df)

    idxs = ~df["is-seen"]
    sns.scatterplot(
        data=df[idxs],
        x="x",
        y="y",
        color="gray",
        style="type",
        style_order=["english", "dutch", "image"],
        legend=True,
        alpha=0.5,
        markers=markers,
        ax=axs[0],
    )
    LEGEND_LABELS = {
        "english": "Audio (English)",
        "dutch": "Audio (Dutch)",
        "french": "Audio (French)",
        "image": "Image",
    }
    handles, labels = axs[0].get_legend_handles_labels()
    new_labels = [LEGEND_LABELS.get(label, label) for label in labels]
    # for h in handles:
    #     h.set_facecolor("black")
    #     h.set_edgecolor("black")
    axs[0].legend(title="", handles=handles, labels=new_labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),)

    idxs = df["is-seen"]
    sns.scatterplot(
        data=df[idxs],
        x="x",
        y="y",
        hue="word-en",
        style="type",
        legend=False,
        markers=markers,
        ax=axs[0],
    )

    def add_texts_2(ax, df, fontsize=10, key=["word", "lang"]):
        def fn(x):
            return x.groupby("modality")[["x", "y"]].mean()

        locs1 = df.groupby(key).apply(fn)
        locs1 = locs1.reset_index()
        locs2 = locs1.groupby("word-en").mean()
        locs2 = locs2.reset_index()
        texts = [
            ax.text(
                row["x"],
                row["y"],
                row["word-en"],
                ha="center",
                va="center",
                size=fontsize,
            )
            for _, row in locs2.iterrows()
        ]
        adjust_text(
            texts,
            # x=df.x.values,
            # y=df.y.values,
            ax=ax,
            force_points=0.5,
            only_move={"points": "y", "explode": "y", "text": "y"},
            # force_text=0.5,
            # lim=15,
            # ha="center",
            # arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
        )
        arrowprops = dict(arrowstyle="->", color="gray", alpha=0.5)
        for text in texts:
            xys = locs1[locs1["word-en"] == text.get_text()]
            xy1 = xys[xys["modality"] == "audio"][["x", "y"]].values
            xy2 = xys[xys["modality"] == "image"][["x", "y"]].values
            ax.annotate(
                text.get_text(),
                xy=xy1,
                xytext=text.get_position(),
                **arrowprops,
            )
            ax.annotate(
                text.get_text(),
                xy=xy2,
                xytext=text.get_position(),
                **arrowprops,
            )

    fs = 20
    clear_axes(axs[0])
    # add_texts_2(axs[0], df[idxs], fontsize=fs, key=["word-en"])
    axs[0].set_title("PCA projection of\naudio and image embeddings")

    # sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    def show_2d(data, embs_2d, modality, langs, ax):
        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["word-en"] = [d["word-en"] for d in data]

        if modality == "audio":
            df["lang"] = [d["lang"] for d in data]
            df["word"] = [translate(d["word-en"], d["lang"]) for d in data]
            kwargs = {
                "style": "lang",
                "markers": {
                    "english": "o",
                    "french": "X",
                    "dutch": "s",
                },
            }
        else:
            kwargs = {
                "marker": "P",
            }

        idxs = df["word-en"].isin(WORDS_SEEN)

        sns.scatterplot(
            data=df[~idxs],
            x="x",
            y="y",
            color="gray",
            alpha=0.5,
            legend=False,
            ax=ax,
            **kwargs,
        )

        sns.scatterplot(
            data=df[idxs],
            x="x",
            y="y",
            hue="word-en",
            legend=False,
            ax=ax,
            **kwargs,
        )

        if modality == "audio":
            add_texts(ax, df[idxs], fontsize=fs)
        else:
            df2 = df.copy()
            df2["word"] = df2["word-en"]
            add_texts(ax, df2[idxs], fontsize=fs, key="word")

        ax.set_title("t-SNE projection of\n{} embeddings".format(modality))
        clear_axes(ax)

    proj = TSNE(n_components=2, random_state=0)
    # proj = PCA(n_components=2, random_state=0)

    show_2d(data_audio, proj.fit_transform(embs_audio), "audio", train_langs, axs[1])
    # show_2d(data_image, proj.fit_transform(embs_image), "image", train_langs, ax3)

    fig.set_tight_layout(True)
    st.pyplot(fig)
    fig.savefig("output/interspeech25/embeddings.pdf")


def evaluate_translation(embeddings_type="out"):
    size = "md"

    def l2_normalize(X):
        return X / np.linalg.norm(X, axis=1)[:, None]

    def get_X_y(data, embs, lang):
        lang_long = LANG_SHORT_TO_LONG[lang]
        idxs = [i for i, datum in enumerate(data) if datum["lang"] == lang_long]
        X = embs[idxs]
        y = [data[i]["word-en"] for i in idxs]
        return X, y

    def eval1(data, embs, lang_src, lang_tgt):
        X_tr, y_tr = get_X_y(data, embs, lang_src)
        X_te, y_te = get_X_y(data, embs, lang_tgt)
        ncm = NearestCentroid()
        ncm.fit(X_tr, y_tr)
        ncm.centroids_ = l2_normalize(ncm.centroids_)
        y_pr = ncm.predict(X_te)
        print(lang_src, lang_tgt)
        print(classification_report(y_te, y_pr))
        print()
        return 100 * ncm.score(X_te, y_te)

    def load_data_and_embs_inp_all(langs, type_):
        langs_long = [LANG_SHORT_TO_LONG[lang] for lang in langs]
        path = f"output/show-model-comparison/audio-data-ss-30.json"
        data = read_json(path)
        data = [
            datum
            for datum in data
            if datum["lang"] in langs_long and datum["word-en"] in WORDS[type_]
        ]
        loader = AudioFeaturesLoader("wavlm-base-plus", "test", langs_long)
        embs = [loader(datum) for datum in data]
        embs = [emb.mean(axis=1) for emb in embs]
        embs = np.stack(embs)
        return data, embs

    def evaluate_embs_inp(train_langs, type_):
        data, embs = load_data_and_embs_inp_all(train_langs, type_)
        return [
            {
                "src": train_langs[0],
                "tgt": train_langs[1],
                "type_": type_,
                "accuracy": eval1(data, embs, train_langs[0], train_langs[1]),
            },
            {
                "src": train_langs[1],
                "tgt": train_langs[0],
                "type_": type_,
                "accuracy": eval1(data, embs, train_langs[1], train_langs[0]),
            },
        ]

    def evaluate_embs_out_1(train_langs, seed, type_):
        def to_keep(datum, modality, langs):
            cond1 = datum["word-en"] in WORDS[type_]
            cond2 = modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs
            return cond1 and cond2

        data, embs = filter(
            *load_data_and_embs_all("audio", train_langs, size, seed),
            partial(to_keep, modality="audio", langs=train_langs),
        )

        return [
            {
                "src": train_langs[0],
                "tgt": train_langs[1],
                "seed": seed,
                "type_": type_,
                "accuracy": eval1(data, embs, train_langs[0], train_langs[1]),
            },
            {
                "src": train_langs[1],
                "tgt": train_langs[0],
                "seed": seed,
                "type_": type_,
                "accuracy": eval1(data, embs, train_langs[1], train_langs[0]),
            },
        ]

    def evaluate_embs_out(train_langs, type_):
        return [evaluate_embs_out_1(train_langs, seed, type_) for seed in "abcde"]

    EVAL_FUNCS = {
        "out": evaluate_embs_out,
        "inp": evaluate_embs_inp,
    }

    eval_func = EVAL_FUNCS[embeddings_type]

    results = [
        r
        for train_langs in [["en", "fr"], ["en", "nl"], ["fr", "nl"]]
        for t in ["seen", "unseen"]
        for r in eval_func(train_langs, t)
    ]
    df = pd.DataFrame(results)
    df = (
        df.groupby(["src", "tgt", "type_"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    # st.write(df.pivot(index=["src", "tgt"], columns="type_").mean())
    df["accuracy"] = (
        df["mean"].round(1).astype(str) + "±" + (2 * df["std"]).round(1).astype(str)
    )
    df = df.drop(columns=["mean", "std"])
    df = df.pivot(index=["src", "tgt"], columns="type_")
    print(df.to_latex())
    st.write(df)


def select_by_word(data, embs, word):
    idxs = [i for i, datum in enumerate(data) if datum["word-en"] == word]
    return embs[idxs]


def compute_var(embs):
    C = np.cov(embs.T)
    return np.trace(C)


def quantify_variance_comparison():
    size = "md"

    def compute1(train_langs, modality, seed, test_langs_idxs):
        def to_keep(datum, modality, langs):
            return modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs

        test_langs = [train_langs[i] for i in test_langs_idxs]
        data, embs = filter(
            *load_data_and_embs_all(modality, train_langs, size, seed),
            partial(to_keep, modality=modality, langs=test_langs),
        )

        def fmt(word, words_type):
            if words_type == "seen":
                return word
            else:
                return word + "'"

        return [
            {
                "word-type": words_type,
                "word": fmt(word, words_type),
                "variance": compute_var(select_by_word(data, embs, word)),
            }
            for words_type in ["seen", "unseen"]
            for word in sorted(WORDS[words_type])
        ]

    modality = "audio"
    results = [
        {
            "train-langs": "-".join(train_langs),
            "test-langs-indices": "+".join(map(str, test_langs_idxs)),
            "seed": seed,
            **result,
        }
        for train_langs in sorted(combinations(["en", "fr", "nl"], 2))
        for test_langs_idxs in [[0], [1], [0, 1]]
        for seed in "abcde"
        for result in compute1(train_langs, modality, seed, test_langs_idxs)
    ]

    df = pd.DataFrame(results)
    st.write(df)

    sns.set(style="whitegrid", font="Arial", context="poster")

    fig = sns.catplot(
        df,
        x="variance",
        y="word",
        hue="test-langs-indices",
        col="train-langs",
        height=11,
    )
    st.pyplot(fig)

    fig = sns.catplot(
        df,
        x="variance",
        y="word-type",
        hue="test-langs-indices",
        col="train-langs",
        kind="box",
    )
    st.pyplot(fig)


def compute_variance_1(train_langs, size, modality, words_type, seed, var_type, test_langs=None):
    def to_keep(datum, modality, langs):
        cond1 = datum["word-en"] in WORDS[words_type]
        cond2 = modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in langs
        return cond1 and cond2

    test_langs = test_langs or train_langs
    data, embs = filter(
        *load_data_and_embs_all(modality, train_langs, size, seed),
        partial(to_keep, modality=modality, langs=train_langs),
    )

    # if modality == "audio" and words_type == "unseen":
    #     dists = np.linalg.norm(embs[:, None] - embs, axis=2)
    #     idxs = np.triu_indices(len(embs), k=1)
    #     dists = dists[idxs]
    #     print(dists.min())
    #     pdb.set_trace()

    if var_type == "full":
        return compute_var(embs)
    elif var_type == "concept":
        word_to_embs = {word: select_by_word(data, embs, word) for word in WORDS[words_type]}
        # word_to_embs = {word: embs[:10] for word, embs in word_to_embs.items()}
        vars = [compute_var(embs1) for embs1 in word_to_embs.values()]
        return np.mean(vars)
    else:
        raise ValueError("Invalid var_type")


def quantify_variance(train_langs=("en", "nl"), use_legend=True):
    size = "md"
    results = [
        {
            "modality": modality,
            "words_type": words_type,
            "seed": seed,
            "var_type": var_type,
            "value": compute_variance_1(
                train_langs, size, modality, words_type, seed, var_type
            ),
        }
        for modality in ["audio", "image"]
        for words_type in ["seen", "unseen"]
        for seed in "abcde"
        for var_type in ["full", "concept"]
    ]

    df = pd.DataFrame(results)
    st.write(df)

    df = df.replace(
        {
            "modality": {"audio": "Audio", "image": "Image"},
            "words_type": {"seen": "Familiar", "unseen": "Novel"},
            "var_type": {"full": "All samples", "concept": "Within class"},
        }
    )

    sns.set(style="whitegrid", font="Arial", context="poster")
    fig = sns.catplot(
        df,
        x="modality",
        y="value",
        hue="words_type",
        col="var_type",
        kind="bar",
        errorbar="sd",
        legend=use_legend,
        # height=8,
        aspect=0.7,
        # legend_out=False,
    )

    if use_legend:
        fig._legend.set_title("")

    for k, ax in fig.axes_dict.items():
        ax.set_title(k)
        ax.set_xlabel("")
        ax.set_ylabel("Variance")
        ax.set_ylim(0.0, 0.3)

    st.pyplot(fig)
    langs_str = "-".join(train_langs)
    fig.savefig(f"output/interspeech25/variance-{langs_str}.pdf")


def quantify_variance_multi():
    def get_results():
        return [
            {
                "train_langs": train_langs,
                "modality": modality,
                "words_type": words_type,
                "seed": seed,
                "var_type": var_type,
                "value": compute_variance_1(
                    train_langs,
                    "md",
                    modality,
                    words_type,
                    seed,
                    var_type,
                ),
            }
            for train_langs in [("en",), ("en", "fr"), ("en", "nl")]
            for modality in ["audio", "image"]
            for words_type in ["seen", "unseen"]
            for seed in "abcde"
            for var_type in ["full", "concept"]
        ]

    results = cache_json("/tmp/variances-multi.json", get_results)
    df = pd.DataFrame(results)
    df = df.replace(
        {
            "modality": {"audio": "Audio", "image": "Image"},
            "words_type": {"seen": "Familiar", "unseen": "Novel"},
            "var_type": {"full": "All samples", "concept": "Within class"},
        }
    )
    # st.write(df)

    kwargs = {
        "x": "modality",
        "y": "value",
        "hue": "words_type",
    }

    def plot1(ax, train_langs, var_type, legend=False):
        train_langs_str = "-".join(train_langs)
        idxs0 = df["train_langs"].str.join("-") == train_langs_str
        idxs1 = df["var_type"] == var_type
        idxs = idxs0 & idxs1
        sns.barplot(data=df[idxs], ax=ax, **kwargs, legend=legend)
        train_langs_fmt_str = ", ".join([lang.upper() for lang in train_langs])
        ax.set_title("Model: {}\n{}".format(train_langs_fmt_str, var_type))
        return ax

    sns.set(style="whitegrid", font="Arial", context="poster")
    fig, axs = plt.subplots(
        1, 4, figsize=(11, 4.75), sharey=True, gridspec_kw={"wspace": 0.0}
    )

    plot1(axs[0], ["en"], "All samples")
    plot1(axs[1], ["en"], "Within class", legend=True)
    plot1(axs[2], ["en", "fr"], "All samples")
    plot1(axs[3], ["en", "nl"], "All samples")

    axs[0].set_ylabel("Variance")

    sns.move_legend(axs[1], "upper center", bbox_to_anchor=(0.5, 1.0), title="", fontsize="small")

    for i in range(4):
        axs[i].set_xlabel("")

    # for i in range(1, 4):
    #     axs[i].set_ylabel("")
    #     axs[i].set_yticklabels([])
    #     axs[i].set_title("")

    fig.set_tight_layout(True)
    st.pyplot(fig)
    # langs_str = "-".join(train_langs)
    fig.savefig(f"output/interspeech25/variance-combined.pdf")


def compute_average_similarity_per_modality():
    size = "md"
    words_type = "seen"

    def compute_average_sim(embs):
        sims = embs @ embs.T
        np.fill_diagonal(sims, 0)
        return {
            "sim-mean": sims.mean(),
            "sim-std": sims.std(),
        }

    def compute1(modality, train_langs, seed):
        def to_keep(datum, modality, selected_langs):
            cond1 = datum["word-en"] in WORDS[words_type]
            cond2 = (
                modality == "image"
                or LANG_LONG_TO_SHORT[datum["lang"]] in selected_langs
            )
            return cond1 and cond2

        data, embs = filter(
            *load_data_and_embs_all(modality, train_langs, size, seed),
            partial(to_keep, modality=modality, selected_langs=train_langs),
        )

        norms = np.linalg.norm(embs, axis=1)
        embs1 = embs / norms[:, None]

        return compute_average_sim(embs1)

    langs = ["en", "fr", "nl"]
    train_langs_combs = [[lang] for lang in langs] + list(combinations(langs, 2))
    results = [
        {
            "modality": modality,
            "words_type": words_type,
            "seed": seed,
            **compute1(modality, train_langs, seed),
        }
        for modality in ["audio", "image"]
        for train_langs in train_langs_combs
        for seed in "abcde"
    ]

    df = pd.DataFrame(results)
    st.write(df)
    st.write(df.groupby(["modality"])["sim-mean", "sim-std"].mean())


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # evaluate_translation("inp")
    # quantify_variance(("en", ), use_legend=False)
    # quantify_variance(("en", "fr"), use_legend=False)
    # quantify_variance(("en", "nl"), use_legend=True)
    quantify_variance_multi()
    # quantify_variance_comparison()
    # compute_average_similarity_per_modality()

    # with st.sidebar:
    #     train_langs = st.multiselect(
    #         "Languages:",
    #         ["en", "fr", "nl"],
    #         default=["en", "nl"],
    #     )
    #     size = st.selectbox("Size:", ["sm", "md", "lg"], index=1)
    #     seed = st.selectbox("Seed:", "abcde", index=2)

    # if len(train_langs) == 0:
    #     st.error("Select at least one language.")
    # else:
    #     do2(tuple(train_langs), size, seed)

    # do3(("en", "nl"), "md", "c")
