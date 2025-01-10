import pdb

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from tbparse import SummaryReader

from mevgs.utils import cache_json
from mevgs.scripts.show_main_table_last_epoch import get_results_seed
from mevgs.scripts.prepare_predictions_for_yevgen import NAMES
from mevgs.scripts.show_3d import DATASET, WORDS_SEEN
from mevgs.scripts.show_crosslingual_audio_alignment import (
    load_data_and_embs_all,
    LANG_SHORT_TO_LONG,
)


WORDS_UNSEEN = DATASET.words_unseen
# WORDS_UNSEEN.remove("nautilus")
WORDS_UNSEEN = [w for w in WORDS_UNSEEN if w != "nautilus"]


def compute_novel_familiar_accuracy(data, embs, selected_word, WORDS_SEEN):
    def is_correct(q, p, n):
        q_emb = embs[q["i"]]
        p_emb = embs[p["i"]]
        n_emb = embs[n["i"]]
        qp = np.linalg.norm(q_emb - p_emb)
        qn = np.linalg.norm(q_emb - n_emb)
        return qp < qn

    data_audio_novel = [
        {**datum, "i": i}
        for i, datum in enumerate(data)
        if datum["modality"] == "audio" and datum["word-en"] == selected_word
    ]
    data_image_novel = [
        {**datum, "i": i}
        for i, datum in enumerate(data)
        if datum["modality"] == "image" and datum["word-en"] == selected_word
    ]
    data_image_familiar = [
        {**datum, "i": i}
        for i, datum in enumerate(data)
        if datum["modality"] == "image" and datum["word-en"] in WORDS_SEEN
    ]

    results = [
        is_correct(q, p, n)
        for q in data_audio_novel[::3]
        for p in data_image_novel[::3]
        for n in data_image_familiar[::3]
    ]
    return 100 * np.mean(results)


def compute_spread(data, embs, selected_word, modality):
    idxs = [
        idx
        for idx, datum in enumerate(data)
        if datum["modality"] == modality and datum["word-en"] == selected_word
    ]
    embs_word = embs[idxs]
    embs_mean = np.mean(embs_word, axis=0)
    embs_mean = embs_mean / np.linalg.norm(embs_mean)
    return np.mean(np.linalg.norm(embs_word - embs_mean, axis=1))


def compute_me(data, embs):
    def compute1(word):
        nf = compute_novel_familiar_accuracy(data, embs, word, WORDS_SEEN)
        return {
            "word": word,
            "me-bias": nf.item(),
        }

    return [compute1(word) for word in WORDS_UNSEEN]


def compute_features(data, embs):
    def compute_spread_1(word, modality):
        spread = compute_spread(data, embs, word, modality)
        return {
            "word": word,
            "value": spread,
        }

    def compute_modality_distance(word):
        idxs_audio = [
            idx
            for idx, datum in enumerate(data)
            if datum["modality"] == "audio" and datum["word-en"] == word
        ]
        idxs_image = [
            idx
            for idx, datum in enumerate(data)
            if datum["modality"] == "image" and datum["word-en"] == word
        ]
        mean_audio = np.mean(embs[idxs_audio], axis=0)
        mean_image = np.mean(embs[idxs_image], axis=0)
        mean_audio = mean_audio / np.linalg.norm(mean_audio)
        mean_image = mean_image / np.linalg.norm(mean_image)
        dist = np.linalg.norm(mean_audio - mean_image)
        return {
            "word": word,
            "value": dist.item(),
        }

    COMPUTE_FEATURES = {
        "spread-image": lambda word: compute_spread_1(word, "image"),
        "spread-audio": lambda word: compute_spread_1(word, "audio"),
        "distance": lambda word: compute_modality_distance(word),
    }

    return [
        {**COMPUTE_FEATURES[feature](word), "feature": feature}
        for word in WORDS_UNSEEN
        for feature in COMPUTE_FEATURES.keys()
    ]


def aggregate_by_word(data, embs):
    words = sorted(set([datum["word-en"] for datum in data]))
    idxss = [
        [i for i, datum in enumerate(data) if datum["word-en"] == word]
        for word in words
    ]
    embs1 = [embs[idxs].mean(0) for idxs in idxss]
    embs1 = np.vstack(embs1)
    embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
    return embs1, words


def filter_data(
    data,
    embs,
    modalities=["image", "audio"],
    words_types=["seen", "unseen"],
):
    WORDS = {
        "seen": WORDS_SEEN,
        "unseen": WORDS_UNSEEN,
    }
    words = [word for words_type in words_types for word in WORDS[words_type]]
    words = set(words)
    idxs = [
        idx
        for idx, datum in enumerate(data)
        if datum["modality"] in modalities and datum["word-en"] in words
    ]
    data1 = [data[idx] for idx in idxs]
    embs1 = embs[idxs]
    return data1, embs1


def compute_seen_word_distances(data, embs):
    def aggregate_by_word_1(embs, data, modality):
        return aggregate_by_word(*filter_data(data, embs, [modality], ["seen"]))

    embs_image, words_image = aggregate_by_word_1(embs, data, "image")
    embs_audio, words_audio = aggregate_by_word_1(embs, data, "audio")
    assert words_image == words_audio
    distances = np.linalg.norm(embs_image[:, np.newaxis] - embs_audio, axis=2)
    return distances


def load_data_and_embs_all_1(train_langs, size, seed, test_lang):
    config = {
        "train_langs": train_langs,
        "size": size,
        "seed": seed,
    }
    image_data, embs_image = load_data_and_embs_all("image", **config)
    audio_data, embs_audio = load_data_and_embs_all("audio", **config)

    test_lang_long = LANG_SHORT_TO_LONG[test_lang]
    idxs = [i for i, datum in enumerate(audio_data) if datum["lang"] == test_lang_long]
    audio_data = [datum for i, datum in enumerate(audio_data) if i in idxs]
    embs_audio = embs_audio[idxs]

    image_data = [{**datum, "modality": "image"} for datum in image_data]
    audio_data = [{**datum, "modality": "audio"} for datum in audio_data]
    data = image_data + audio_data
    embs = np.concatenate([embs_image, embs_audio], axis=0)

    return data, embs


def cumulative_residual_variance_1(Y, X):
    """
    Y has shape (N, D)
    X has shape (M, D)
    """

    def compute_var(X):
        return np.trace(np.cov(X))

    Y1 = Y.copy()
    Y1 = Y1 - np.mean(Y1, axis=0, keepdims=True)

    var0 = compute_var(Y1)
    vars_residual = [var0]

    pca = PCA()
    pca.fit(X)

    for v in pca.components_:
        proj = Y1 @ v
        Y1 = Y1 - proj[:, None] @ v[None, :]
        vars_residual.append(compute_var(Y1))

    vars_residual = np.array(vars_residual) / var0
    vars_explained = np.cumsum(pca.explained_variance_ratio_)
    vars_explained = np.concatenate(
        [
            np.array(
                [
                    0,
                ]
            ),
            vars_explained,
        ]
    )
    return vars_residual, vars_explained


def cumulative_residual_variance_2(y_vecs, x_vecs):
    """computes CRV(X/Y)"""

    def residual_variance_ratio(vecs, pca_components):
        """
        the amount of variance remaining in vecs
        after collapsing each of the principal components of pca
        """
        vecs_n = np.array(vecs) - np.mean(vecs, axis=0)
        residual_variance = []
        pca_vecs = PCA(n_components=min(vecs_n.shape[0], vecs_n.shape[1]))
        # print(pca_vecs)
        pca_vecs.fit(vecs)
        residual_variance.append(sum(pca_vecs.explained_variance_))
        pdb.set_trace()
        for pc in pca_components:
            proj = np.dot(np.array(vecs_n), pc)
            collapsed_vecs = np.array(vecs_n) - np.dot(proj[:, None], pc[None,])
            pca_collapsed = PCA(n_components=min(vecs_n.shape[0], vecs_n.shape[1]))
            pca_collapsed.fit(collapsed_vecs)
            residual_variance.append(sum(pca_collapsed.explained_variance_))
            vecs_n = collapsed_vecs
        # collapsed_variance = np.concatenate([np.array([0,]),np.cumsum(pca.explained_variance_ratio_)])
        return (
            np.array(residual_variance) / residual_variance[0]
        )  # , collapsed_variance

    # y_pca = PCA(n_components=len(y_vecs))
    # y_pca.fit(y_vecs)
    y_vecs = y_vecs.T
    x_vecs = x_vecs.T
    residual_var = residual_variance_ratio(y_vecs, x_vecs)
    return residual_var


def do1(train_langs, size, seed, test_lang):
    data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)

    train_langs_str = "-".join(train_langs) if train_langs else "random"
    config_name = f"{train_langs_str}_{size}_{seed}_{test_lang}"
    path = f"/tmp/me-{config_name}.json"
    data_me = cache_json(path, compute_me, data, embs)
    data_feats = compute_features(data, embs)

    features = sorted(set([datum["feature"] for datum in data_feats]))

    # dists = compute_seen_word_distances(data, embs)
    # mean_diag = np.mean(np.diag(dists))
    # mean_offdiag = np.mean(dists[~np.eye(dists.shape[0], dtype=bool)])
    # st.write(mean_diag, mean_offdiag)

    # def get_eigs(data, embs, modality, words_type):
    #     embs = aggregate_by_word(*filter_data(data, embs, [modality], [words_type]))
    #     N = len(embs)
    #     C = np.cov(embs.T)
    #     D, eigs = np.linalg.eig(C)
    #     return eigs[:, :N].real

    # def compute_coeffs_unseen_to_seen(data, embs, modality, ax):
    #     eigs_seen = get_eigs(data, embs, modality, "seen")
    #     eigs_unseen = get_eigs(data, embs, modality, "unseen")
    #     cos_sim = eigs_seen.T @ eigs_unseen
    #     # sns.heatmap(cos_sim, ax=ax, square=True, cbar=False, vmin=0, vmax=1)

    #     _, embs_unseen = filter_data(data, embs, [modality], ["unseen"])
    #     _, embs_seen = filter_data(data, embs, [modality], ["seen"])
    #     # X = np.random.rand(100, 10)
    #     res, exp = cumulative_residual_variance_1(embs_unseen, embs_seen)
    #     # res, exp = cumulative_residual_variance_1(X, X)
    #     ax.plot(exp, res)
    #     ax.set_xlabel("Explained variance")
    #     ax.set_ylabel("Residual variance")
    #     ax.set_title("AUC = {:.2f}".format(np.trapz(res, exp)))
    #     # vars = cumulative_residual_variance_1(eigs_unseen, eigs_seen)
    #     # st.write(vars)
    #     # st.write(cumulative_residual_variance_2(eigs_unseen, eigs_seen))

    #     # coeffs = np.linalg.pinv(embs_unseen.T) @ embs_seen.T
    #     # st.write(coeffs)
    #     # pdb.set_trace()

    # fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharey=True)
    # compute_coeffs_unseen_to_seen(data, embs, "audio", axs[0])
    # compute_coeffs_unseen_to_seen(data, embs, "image", axs[1])
    # st.pyplot(fig)

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), sharey=True)
    for i, feature in enumerate(features):
        df1 = pd.DataFrame(
            [datum for datum in data_feats if datum["feature"] == feature]
        )
        df2 = pd.DataFrame(data_me)
        df = pd.merge(df1, df2, on="word")
        corr = 100 * np.corrcoef(df["value"], df["me-bias"])[0, 1]
        sns.scatterplot(data=df, x="value", y="me-bias", ax=axs[i])
        axs[i].set_xlabel(feature)
        axs[i].set_title("ρ: {:.1f}%".format(corr))
        axs[i].axhline(50, color="red", linewidth=1, linestyle="--")

    st.markdown(
        "Train langs: {} · Size: {} · Seed: {} · Test lang: {}".format(
            train_langs_str, size, seed, test_lang
        )
    )
    st.pyplot(fig)


def show_across_words_mismatched():
    import random
    random.seed(42)

    def compute_me_mismatched_pair(data, embs, word_query, word_positive):
        def is_correct(q, p, n):
            q_emb = embs[q["i"]]
            p_emb = embs[p["i"]]
            n_emb = embs[n["i"]]
            qp = np.linalg.norm(q_emb - p_emb)
            qn = np.linalg.norm(q_emb - n_emb)
            return qp < qn

        data_audio_novel = [
            {**datum, "i": i}
            for i, datum in enumerate(data)
            if datum["modality"] == "audio" and datum["word-en"] == word_query
        ]
        data_image_novel = [
            {**datum, "i": i}
            for i, datum in enumerate(data)
            if datum["modality"] == "image" and datum["word-en"] == word_positive
        ]
        data_image_familiar = [
            {**datum, "i": i}
            for i, datum in enumerate(data)
            if datum["modality"] == "image" and datum["word-en"] in WORDS_SEEN
        ]

        def sample_pos_neg():
            p = random.choice(data_image_novel)
            negs = [n for n in data_image_familiar if n["source"] == p["source"]]
            n = random.choice(negs)
            return p, n

        results = [
            is_correct(q, *sample_pos_neg())
            for q in data_audio_novel
            for _ in range(30)
        ]
        return {
            "me-bias": 100 * np.mean(results),
            "word-query": word_query,
            "word-pos": word_positive,
        }

    def compute_me_mismatched(data, embs):
        return [
            compute_me_mismatched_pair(data, embs, q, p)
            for q, p in tqdm(product(WORDS_UNSEEN, WORDS_UNSEEN))
        ]

    def compute_distances(data, embs):
        audio_data, embs_audio = filter_data(data, embs, ["audio"], ["unseen"])
        image_data, embs_image = filter_data(data, embs, ["image"], ["unseen"])

        audio_centroids, audio_words = aggregate_by_word(audio_data, embs_audio)
        image_centroids, image_words = aggregate_by_word(image_data, embs_image)

        audio_centroids = dict(zip(audio_words, audio_centroids))
        image_centroids = dict(zip(image_words, image_centroids))

        def compute_modality_distance(q, p):
            dist = np.linalg.norm(audio_centroids[q] - image_centroids[p])
            return {
                "distance": dist.item(),
                "word-query": q,
                "word-pos": p,
            }

        return [
            compute_modality_distance(q, p)
            for q, p in tqdm(product(WORDS_UNSEEN, WORDS_UNSEEN))
        ]

    def do1(train_langs, size, seed, test_lang, ax, use_legend=False):
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)

        train_langs_str = "-".join(train_langs) if train_langs else "random"
        config_name = f"{train_langs_str}_{size}_{seed}_{test_lang}"

        path = f"output/show-explain-me/me-mismatched-{config_name}.json"
        data_me = cache_json(path, compute_me_mismatched, data, embs)
        df_me = pd.DataFrame(data_me)

        path = f"output/show-explain-me/dists-mismatched-{config_name}.json"
        data_dists = cache_json(path, compute_distances, data, embs)
        df_dists = pd.DataFrame(data_dists)

        df = pd.merge(df_me, df_dists, on=["word-query", "word-pos"])
        df["is-paired"] = df["word-query"] == df["word-pos"]

        corr = 100 * np.corrcoef(df["distance"], df["me-bias"])[0, 1]
        # me_bias_mean = df["me-bias"].mean()
        idxs = df["is-paired"]
        me_bias_mean = df[idxs]["me-bias"].mean()
        me_bias_model = get_results_seed(train_langs, "no", size, test_lang, seed)["NF"]

        xlabel = "Distance between\naudio–image centroids"
        ylabel = "Novel–familiar accuracy"
        model_str = "{}: {}".format(
            "Monolingual" if len(train_langs) == 1 else "Bilingual",
            ", ".join([LANG_SHORT_TO_LONG[lang].capitalize() for lang in train_langs]),
        )

        df = df.rename(
            columns={
                "word-query": "Query",
                "me-bias": ylabel,
                "distance": xlabel,
            }
        )
        df["Pair type"] = df["is-paired"].map({False: "NN'", True: "NN"})

        st.markdown(
            "Train langs: {} · Size: {} · Seed: {} · Test lang: {} · ME computed: {:.1f}% · ME model: {:.1f}%".format(
                train_langs_str,
                size,
                seed,
                test_lang,
                me_bias_mean,
                me_bias_model,
            )
        )
        sns.scatterplot(
            data=df,
            x=xlabel,
            y=ylabel,
            ax=ax,
            hue="Query",
            style="Pair type",
            markers={"NN'": "X", "NN": "o"},
            legend=use_legend,
        )
        ax.set_title("{}\nNF: {:.1f}%".format(model_str, me_bias_model))
        ax.axhline(50, color="black", linewidth=1, linestyle="--")

    # train_langs = ["en"]
    # seed = "c"
    # size = "sm"
    # test_lang = "en"
    # do1(train_langs, size, seed, test_lang)

    sns.set(style="whitegrid", font="Arial", context="poster")
    fig, axs = plt.subplots(figsize=(12, 6), ncols=2, nrows=1, sharey=True)
    do1(["en"], "md", "c", "en", axs[0])
    do1(["en", "nl"], "md", "d", "en", axs[1], use_legend=True)
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1), ncols=2, fontsize=14)

    fig.set_tight_layout(True)
    st.pyplot(fig)

    fig.savefig("output/interspeech25/me-vs-distance.pdf")


def show_across_words():
    train_langs = ["en"]
    seed = "a"
    # for train_langs in [["en"], ["en", "fr"], ["en", "nl"]]:
    for size in "sm md lg".split():
        # for seed in "abcde":
        do1(train_langs, size, seed, "en")
    # for seed in "abcde":
    #     do1(None, "sm", seed, "en")


def show_across_models(test_lang="en"):
    WORDS = {
        "seen": WORDS_SEEN,
        "unseen": WORDS_UNSEEN,
    }

    def compute_spread(data, embs, modality, words_type):
        idxs = [
            idx
            for idx, datum in enumerate(data)
            if datum["modality"] == modality and datum["word-en"] in WORDS[words_type]
        ]
        embs_word = embs[idxs]
        embs_mean = np.mean(embs_word, axis=0)
        embs_mean = embs_mean / np.linalg.norm(embs_mean)
        return np.mean(np.linalg.norm(embs_word - embs_mean, axis=1))

    def compute_modality_distance_seen(data, embs):
        # _, embs_image = filter_data(data, embs, ["image"], ["seen"])
        # _, embs_audio = filter_data(data, embs, ["audio"], ["seen"])
        # embs_image_centroid = np.mean(embs_image, axis=0)
        # embs_audio_centroid = np.mean(embs_audio, axis=0)
        embs_image, words_image = aggregate_by_word(
            *filter_data(data, embs, ["image"], ["seen"])
        )
        embs_audio, words_audio = aggregate_by_word(
            *filter_data(data, embs, ["audio"], ["seen"])
        )
        assert words_image == words_audio
        dists = np.linalg.norm(embs_image - embs_audio, axis=1)
        return np.mean(dists)

    def compute_modality_distance_unseen(data, embs):
        _, embs_image = filter_data(data, embs, ["image"], ["unseen"])
        _, embs_audio = filter_data(data, embs, ["audio"], ["unseen"])
        dists = pairwise_distances(embs_image, embs_audio)
        return np.mean(dists)
        # embs_image, words_image = aggregate_by_word(*filter_data(data, embs, ["image"], ["unseen"]))
        # embs_audio, words_audio = aggregate_by_word(*filter_data(data, embs, ["audio"], ["unseen"]))
        # assert words_image == words_audio
        # dists = np.linalg.norm(embs_image - embs_audio, axis=1)
        # return np.mean(dists)

    def get_result(train_langs, size, seed, test_lang):
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)
        result = {
            f"spread-{m}-{t}": compute_spread(data, embs, m, t)
            for m in ["image", "audio"]
            for t in ["seen", "unseen"]
        }
        return {
            **result,
            "dist-unseen": compute_modality_distance_unseen(data, embs),
            "dist-seen": compute_modality_distance_seen(data, embs),
            "me-bias": get_results_seed(train_langs, "no", size, test_lang, seed)["NF"],
        }

    LANGS = ["en", "fr", "nl"]
    train_langs_combinations = [tuple(sorted(set([test_lang, lang]))) for lang in LANGS]
    train_langs_combinations = sorted(train_langs_combinations, key=lambda x: len(x))

    results = [
        {
            **get_result(train_langs, size, seed, test_lang),
            "size": size,
            "seed": seed,
            "train-langs": "-".join(train_langs),
        }
        for train_langs in train_langs_combinations
        # for size in "sm md lg".split()
        for size in ["md"]
        for seed in "abcde"
    ]

    df = pd.DataFrame(results)

    df["spread-image-ratio"] = df["spread-image-unseen"] / df["spread-image-seen"]
    df["spread-audio-ratio"] = df["spread-audio-unseen"] / df["spread-audio-seen"]
    df["dist-ratio"] = df["dist-unseen"] / df["dist-seen"]

    # st.write(df)
    st.write(df.groupby(["size", "train-langs"]).mean())
    df.to_csv("output/show-explain-me/me-bias-across-models.csv", index=False)

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(14, 4), sharey=True)
    for i, x in enumerate(["spread-image-ratio", "spread-audio-ratio", "dist-ratio"]):
        sns.scatterplot(
            data=df,
            x=x,
            y="me-bias",
            hue="size",
            style="train-langs",
            ax=axs[i],
            legend=(i == 2),
        )
        corr = 100 * np.corrcoef(df[x], df["me-bias"])[0, 1]
        axs[i].set_title("ρ: {:.1f}%".format(corr))

    sns.move_legend(axs[2], "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


def show_across_models_nn_vs_nf(test_lang):
    size = "md"
    # seed = "c"
    # test_lang = "en"

    LANGS = ["en", "fr", "nl"]
    train_langs_combinations = [tuple(sorted(set([test_lang, lang]))) for lang in LANGS]
    train_langs_combinations = sorted(train_langs_combinations, key=lambda x: len(x))

    def compute_modality_distance(
        data,
        embs,
        words_audio="unseen",
        words_image="unseen",
    ):
        _, embs_audio = filter_data(data, embs, ["audio"], [words_audio])
        _, embs_image = filter_data(data, embs, ["image"], [words_image])
        dists = pairwise_distances(embs_image, embs_audio)
        dists = dists.flatten()[::3]
        return dists

    def do1(train_langs, seed):
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)
        return [
            {"distance": d, "pair-type": f"unseen–{i}"}
            for i in ["seen", "unseen"]
            for d in compute_modality_distance(data, embs, i)
        ]

    data = [
        {**datum, "train-langs": "-".join(train_langs), "seed": seed}
        for train_langs in train_langs_combinations
        for seed in "abcde"
        for datum in do1(train_langs, seed)
    ]
    df = pd.DataFrame(data)

    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.stripplot(df, x="distance", y="pair-type", hue="train-langs", ax=ax, color=".3", size=4)

    fig = sns.catplot(
        df,
        kind="box",
        x="distance",
        col="train-langs",
        hue="pair-type",
        y="seed",
    )
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


def show_across_models_nn_vs_nf_2():
    sns.set(style="whitegrid", font="Arial", context="poster")

    size = "md"
    test_lang = "en"
    seed = "c"

    LANGS = ["en", "fr", "nl"]
    train_langs_combinations = [tuple(sorted(set([test_lang, lang]))) for lang in LANGS]
    train_langs_combinations = sorted(train_langs_combinations, key=lambda x: len(x))

    def compute_distances_all(embs1, embs2, *args):
        dists = pairwise_distances(embs1, embs2)
        return dists.flatten()

    def compute_distances_matched(embs1, embs2, data1, data2):
        def get_idxs(data, word):
            idxs = [i for i, datum in enumerate(data) if datum["word-en"] == word]
            return np.array(idxs)

        def compute_distances_all_(word):
            idxs1 = get_idxs(data1, word)
            idxs2 = get_idxs(data2, word)
            return compute_distances_all(embs1[idxs1], embs2[idxs2])

        words = sorted(set([datum["word-en"] for datum in data1]))
        dists = [compute_distances_all_(word) for word in words]
        return np.concatenate(dists)

    PAIR_TYPES = {
        "NF": {
            "words_audio": "seen",
            "words_image": "unseen",
            "compute_distances": compute_distances_all,
        },
        "NN": {
            "words_audio": "unseen",
            "words_image": "unseen",
            "compute_distances": compute_distances_matched,
        },
        "NN'": {
            "words_audio": "unseen",
            "words_image": "unseen",
            "compute_distances": compute_distances_all,
        },
    }

    def compute_modality_distance(
        data,
        embs,
        words_audio="unseen",
        words_image="unseen",
        compute_distances=compute_distances_all,
    ):
        data_audio, embs_audio = filter_data(data, embs, ["audio"], [words_audio])
        data_image, embs_image = filter_data(data, embs, ["image"], [words_image])
        return compute_distances(embs_audio, embs_image, data_audio, data_image)

    def do1(train_langs, seed):
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)
        return [
            {"distance": d, "pair-type": p}
            for p in PAIR_TYPES
            for d in compute_modality_distance(data, embs, **PAIR_TYPES[p])
        ]

    data = [
        {**datum, "train-langs": "-".join(train_langs), "seed": seed}
        for train_langs in train_langs_combinations
        # for seed in "abcde"
        for datum in do1(train_langs, seed)
    ]
    df = pd.DataFrame(data)

    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.stripplot(df, x="distance", y="pair-type", hue="train-langs", ax=ax, color=".3", size=4)

    def map_langs(lang_str):
        langs_long = [LANG_SHORT_TO_LONG[lang] for lang in lang_str.split("-")]
        langs_long = [lang.capitalize() for lang in langs_long]
        return ", ".join(langs_long)

    fig = sns.catplot(
        df,
        kind="box",
        x="distance",
        # row="seed",
        col="train-langs",
        y="pair-type",
        height=4,
    )

    for ax in fig.axes.flat:
        _, langs_str = ax.get_title().split(" = ")
        ax.set_title(map_langs(langs_str))
        ax.set_xlabel("Distance")
        ax.set_ylabel("")
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    fig.savefig("output/interspeech25/nf-nn-nn1.pdf")


def show_across_models_scaling_translation():
    def compute_variance(embs):
        return np.trace(np.cov(embs.T))

    def scale_embs(data, embs, α):
        μ = np.mean(embs, axis=0)
        embs1 = np.sqrt(α) * (embs - μ) + μ
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data, embs1

    def translate_embs(data, embs, γ):
        pca = PCA()
        pca.fit(embs)
        embs1 = embs + γ * pca.components_[0]
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data, embs1

    def transform_embs_all(data, embs, α, γ):
        data_audio_unseen, embs_audio_unseen = filter_data(
            data, embs, ["audio"], ["unseen"]
        )
        data_image_unseen, embs_image_unseen = filter_data(
            data, embs, ["image"], ["unseen"]
        )
        data_audio_seen, embs_audio_seen = filter_data(data, embs, ["audio"], ["seen"])
        data_image_seen, embs_image_seen = filter_data(data, embs, ["image"], ["seen"])

        # var_audio_unseen = compute_variance(embs_audio_unseen)
        # var_image_unseen = compute_variance(embs_image_unseen)
        # var_audio_seen = compute_variance(embs_audio_seen)
        # var_image_seen = compute_variance(embs_image_seen)

        data_audio_unseen, embs_audio_unseen = scale_embs(
            data_audio_unseen, embs_audio_unseen, α
        )
        data_image_unseen, embs_image_unseen = scale_embs(
            data_image_unseen, embs_image_unseen, α
        )

        ma0 = embs_audio_unseen.mean(0)
        mi0 = embs_image_unseen.mean(0)

        data_audio_unseen, embs_audio_unseen = translate_embs(
            data_audio_unseen, embs_audio_unseen, γ
        )
        data_image_unseen, embs_image_unseen = translate_embs(
            data_image_unseen, embs_image_unseen, γ
        )

        ma1 = embs_audio_unseen.mean(0)
        mi1 = embs_image_unseen.mean(0)

        print(np.linalg.norm(ma0 - ma1))
        print(np.linalg.norm(mi0 - mi1))

        data1 = (
            data_audio_unseen + data_image_unseen + data_audio_seen + data_image_seen
        )
        embs1 = np.concatenate(
            [embs_audio_unseen, embs_image_unseen, embs_audio_seen, embs_image_seen],
            axis=0,
        )

        # β1_audio = var_audio_seen / var_audio_unseen
        # β1_image = var_image_seen / var_image_unseen

        # α1 = (α1_audio + α1_image) / 2
        # β1 = (β1_audio + β1_image) / 2
        return data1, embs1

    def compute1(train_langs, size, seed, test_lang, α, γ):
        print(train_langs, size, seed, test_lang, α, γ)
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)
        data, embs = transform_embs_all(data, embs, α, γ)
        return {
            # "α1": α1,
            # "β1": β1,
            "word-me-biases": compute_me(data, embs),
        }

    def get_path(train_langs, size, seed, test_lang, α, γ):
        train_langs_str = "-".join(train_langs) if train_langs else "random"
        config_name = f"{train_langs_str}_{size}_{seed}_{test_lang}_{α}_{γ}"
        return f"output/show-explain-me/me-scaling-translation-{config_name}.json"

    train_langs = ["en"]
    test_lang = "en"
    results = [
        {
            "size": size,
            "seed": seed,
            "α": α,
            "γ": γ,
            **cache_json(
                get_path(train_langs, size, seed, test_lang, α, γ),
                compute1,
                train_langs,
                size,
                seed,
                test_lang,
                α,
                γ,
            ),
        }
        for size in ["sm", "md", "lg"]
        for seed in "abcde"
        # for α in [0.01, 0.1, 0.5, 0.9, 1, 1.1, 2, 5, 10]
        for α in [1]
        for γ in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    ]

    for datum in results:
        datum["me-bias"] = np.mean([d["me-bias"] for d in datum["word-me-biases"]])

    df = pd.DataFrame(results)
    # st.write(df)
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(5, 4))
    sns.scatterplot(data=df, x="γ", y="me-bias", hue="size", ax=ax)
    sns.lineplot(
        data=df, x="γ", y="me-bias", hue="size", units="seed", estimator=None, ax=ax
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # β = df["β1"].mean()
    # ax.axvline(β, color="blue", linewidth=1, linestyle="--")
    # ax.axhline(50, color="red", linewidth=1, linestyle="--")
    # ax.set_xlabel("scaling factor variance unseen")
    st.pyplot(fig)


def show_across_models_random():
    def compute_variance(embs):
        return np.trace(np.cov(embs.T))

    def generate_random(data, embs):
        mu = np.mean(embs, axis=0)
        cov = np.cov(embs.T)
        embs1 = np.random.multivariate_normal(mu, cov, len(embs))
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data, embs1

    def generate_none(data_unseen, embs_unseen, data_seen, embs_seen):
        return data_unseen, embs_unseen

    def generate_mean_unseen_cov_unseen(data_unseen, embs_unseen, data_seen, embs_seen):
        mu = np.mean(embs_unseen, axis=0)
        cov = np.cov(embs_unseen.T)
        embs1 = np.random.multivariate_normal(mu, cov, len(embs_unseen))
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data_unseen, embs1

    def generate_mean_seen_cov_unseen(data_unseen, embs_unseen, data_seen, embs_seen):
        mu = np.mean(embs_seen, axis=0)
        cov = np.cov(embs_unseen.T)
        embs1 = np.random.multivariate_normal(mu, cov, len(embs_unseen))
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data_unseen, embs1

    def generate_mean_seen_cov_seen_scaled(
        data_unseen, embs_unseen, data_seen, embs_seen
    ):
        mu_seen = np.mean(embs_seen, axis=0)
        cov_seen = np.cov(embs_seen.T)

        mu_unseen = np.mean(embs_unseen, axis=0)
        cov_unseen = np.cov(embs_unseen.T)

        var_seen = np.trace(cov_seen)
        var_unseen = np.trace(cov_unseen)

        α = var_unseen / var_seen
        embs1 = np.random.multivariate_normal(mu_unseen, α * cov_seen, len(embs_unseen))

        print(var_unseen)
        print(compute_variance(embs1))

        pdb.set_trace()

        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data_unseen, embs1

    def generate_mean_seen_cov_seen(data_unseen, embs_unseen, data_seen, embs_seen):
        mu = np.mean(embs_seen, axis=0)
        cov = np.cov(embs_seen.T)
        embs1 = np.random.multivariate_normal(mu, cov, len(embs_unseen))
        embs1 = embs1 / np.linalg.norm(embs1, axis=1)[:, np.newaxis]
        return data_unseen, embs1

    def generate_all(data, embs, generate_type):
        data_audio_unseen, embs_audio_unseen = filter_data(
            data, embs, ["audio"], ["unseen"]
        )
        data_image_unseen, embs_image_unseen = filter_data(
            data, embs, ["image"], ["unseen"]
        )

        data_audio_seen, embs_audio_seen = filter_data(data, embs, ["audio"], ["seen"])
        data_image_seen, embs_image_seen = filter_data(data, embs, ["image"], ["seen"])

        # data_audio_unseen, embs_audio_unseen = generate_random(data_audio_unseen, embs_audio_unseen)
        # data_image_unseen, embs_image_unseen = generate_random(data_image_unseen, embs_image_unseen)
        generate_random = GENERATE_TYPES[generate_type]
        data_audio_unseen, embs_audio_unseen = generate_random(
            data_audio_unseen, embs_audio_unseen, data_audio_seen, embs_audio_seen
        )
        data_image_unseen, embs_image_unseen = generate_random(
            data_image_unseen, embs_image_unseen, data_image_seen, embs_image_seen
        )

        data1 = (
            data_audio_unseen + data_image_unseen + data_audio_seen + data_image_seen
        )
        embs1 = np.concatenate(
            [embs_audio_unseen, embs_image_unseen, embs_audio_seen, embs_image_seen],
            axis=0,
        )

        return data1, embs1

    def compute1(train_langs, size, seed, test_lang, generate_type):
        print(train_langs, size, seed, test_lang, generate_type)
        data, embs = load_data_and_embs_all_1(train_langs, size, seed, test_lang)
        data, embs = generate_all(data, embs, generate_type)
        return {
            "word-me-biases": compute_me(data, embs),
        }

    def get_path(train_langs, size, seed, test_lang, generate_type):
        train_langs_str = "-".join(train_langs) if train_langs else "random"
        config_name = f"{train_langs_str}_{size}_{seed}_{test_lang}_{generate_type}"
        return f"output/show-explain-me/me-random-{config_name}.json"

    GENERATE_TYPES = {
        "none": generate_none,
        "μ ← unseen · Σ ← unseen": generate_mean_unseen_cov_unseen,
        "μ ← seen · Σ ← unseen": generate_mean_seen_cov_unseen,
        "μ ← seen · Σ ← α seen": generate_mean_seen_cov_seen_scaled,
        "μ ← seen · Σ ← seen": generate_mean_seen_cov_seen,
    }

    train_langs = ["en"]
    test_lang = "en"
    results = [
        {
            "size": size,
            "seed": seed,
            "sampling": generate_type,
            **cache_json(
                get_path(train_langs, size, seed, test_lang, generate_type),
                compute1,
                train_langs,
                size,
                seed,
                test_lang,
                generate_type,
            ),
        }
        for size in ["sm", "md", "lg"]
        for seed in "abcde"
        for generate_type in GENERATE_TYPES
    ]

    for datum in results:
        word_me_biases = datum.pop("word-me-biases")
        datum["me-bias"] = np.mean([d["me-bias"] for d in word_me_biases])

    df = pd.DataFrame(results)
    st.write(df)
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(5, 4))
    # sns.scatterplot(data=df, x="γ", y="me-bias", hue="size", ax=ax)
    # sns.lineplot(data=df, x="γ", y="me-bias", hue="size", units="seed", estimator=None, ax=ax)
    # fig = sns.catplot(data=df, x="size", y="me-bias", col="to-generate")

    sns.stripplot(data=df, x="me-bias", y="sampling", hue="size", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # show_across_words()
    show_across_words_mismatched()
    # for t in ["en", "fr", "nl"]:
    #     show_across_models_nn_vs_nf(t)
    #     show_across_models(t)
    # show_across_models_nn_vs_nf_2()
    # show_across_models_scaling_translation()
    # show_across_models_random()
