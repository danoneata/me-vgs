import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D
from tbparse import SummaryReader

from mevgs.utils import cache_np, read_json, mapt
from mevgs.predict import load_model, CONFIGS
from mevgs.scripts.show_crosslingual_audio_alignment import DATASET, WORDS_SEEN
from mevgs.scripts.show_model_comparison import (
    compute_image_embeddings_given_model,
    compute_audio_embeddings_given_model,
)


def make_rotation_matrix(α, β, γ):
    α = np.deg2rad(α)
    β = np.deg2rad(β)
    γ = np.deg2rad(γ)
    R1 = np.array([
        [np.cos(α), -np.sin(α), 0],
        [np.sin(α), np.cos(α), 0],
        [0, 0, 1],
    ])
    R2 = np.array([
        [np.cos(β), 0, np.sin(β)],
        [0, 1, 0],
        [-np.sin(β), 0, np.cos(β)],
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, np.cos(γ), -np.sin(γ)],
        [0, np.sin(γ), np.cos(γ)],
    ])
    return R1 @ R2 @ R3


def make_sphere(α=1.0):
    n_points = 1000
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = α * np.outer(np.cos(u), np.sin(v))
    y = α * np.outer(np.sin(u), np.sin(v))
    z = α * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def load_model_and_config(config_name):
    config = CONFIGS[config_name]
    model = load_model(config_name, config)
    model.to(config["device"])
    return model, config


def compute_image_embeddings(config_name, image_data):
    model, config = load_model_and_config(config_name)
    return compute_image_embeddings_given_model(model, config, image_data)


def compute_audio_embeddings(config_name, audio_data):
    model, config = load_model_and_config(config_name)
    return compute_audio_embeddings_given_model(model, config, audio_data)


def load_data_and_embs_image(config_name):
    path = f"output/show-model-comparison/image-data-ss-30.json"
    image_data = read_json(path)

    path = f"output/show-3d/embeddings-image-{config_name}.npy"
    embs = cache_np(path, compute_image_embeddings, config_name, image_data)

    # Filter by language and seen words
    idxs_image_data = [
        (idx, datum)
        for idx, datum in enumerate(image_data)
        # if datum["word-en"] in WORDS_SEEN
    ]
    idxs, image_data = mapt(list, zip(*idxs_image_data))
    embs = embs[idxs]

    return image_data, embs


def load_data_and_embs_audio(config_name):
    langs_long = [
        "english",
        # "french",
    ]
    path = f"output/show-model-comparison/audio-data-ss-30.json"
    audio_data = read_json(path)

    path = f"output/show-3d/embeddings-audio-{config_name}.npy"
    embs = cache_np(path, compute_audio_embeddings, config_name, audio_data)

    # Filter by language and seen words
    idxs_audio_data = [
        (idx, datum)
        for idx, datum in enumerate(audio_data)
        if datum["lang"] in langs_long
        # and datum["word-en"] in WORDS_SEEN
    ]
    idxs, audio_data = mapt(list, zip(*idxs_audio_data))
    embs = embs[idxs]

    return audio_data, embs


def get_result(path):
    results = SummaryReader(path)
    scalars = results.scalars
    col1 = "valid/loss"

    test_lang_long = "english"
    col_nf = f"test/{test_lang_long}/accuracy-novel-familiar"
    if col_nf not in scalars["tag"].unique():
        col_nf = f"test/accuracy-novel-familiar"

    col_ff = f"test/{test_lang_long}/accuracy-familiar-familiar"
    if col_ff not in scalars["tag"].unique():
        col_ff = f"test/accuracy-familiar-familiar"

    df = scalars[scalars["tag"].isin([col1, col_nf, col_ff])]
    df = df.pivot(index="step", columns="tag", values="value")
    best_epoch = df[col1].idxmin()

    return {
        "epoch-best": best_epoch,
        "NF": 100 * df.loc[best_epoch, col_nf],
        "NF-best": 100 * df[col_nf].max(),
        "NF-last": 100 * df.loc[24, col_nf],
        "FF": 100 * df.loc[best_epoch, col_ff],
        "FF-last": 100 * df.loc[24, col_ff],
    }


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    configs = "en_3d en-fr_3d en-nl_3d".split()

    with st.sidebar:
        config_name = st.selectbox("Model:", configs)
        selected_word = st.selectbox("Word:", sorted(DATASET.words_unseen))
        st.markdown("---")
        α = st.number_input("yaw angle", value=0.0, step=10.0, min_value=0.0, max_value=360.0)
        β = st.number_input("pitch angle", value=0.0, step=10.0, min_value=0.0, max_value=360.0)
        γ = st.number_input("roll angle", value=0.0, step=10.0, min_value=0.0, max_value=360.0)

    results = get_result(f"output/{config_name}")
    st.write(results)

    image_data, embs_image = load_data_and_embs_image(config_name)
    audio_data, embs_audio = load_data_and_embs_audio(config_name)

    image_data = [{**datum, "modality": "image"} for datum in image_data]
    audio_data = [{**datum, "modality": "audio"} for datum in audio_data]
    data = image_data + audio_data
    embs = np.concatenate([embs_image, embs_audio], axis=0)

    WORD_TO_IDX = {word: idx for idx, word in enumerate(WORDS_SEEN)}
    NUM_WORDS = len(WORDS_SEEN)

    MODALITY_TO_MARKER = {
        "image": "o",
        "audio": "^",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})

    R = make_rotation_matrix(α, β, γ)
    embs1 = embs @ R

    for m in ["audio", "image"]:
        idxs = [idx for idx, datum in enumerate(data) if datum["modality"] == m and datum["word-en"] in WORDS_SEEN]
        colors = [WORD_TO_IDX[datum["word-en"]] for datum in data if datum["modality"] == m and datum["word-en"] in WORDS_SEEN]
        marker = MODALITY_TO_MARKER[m]
        xs, ys, zs = embs1[idxs].T
        ax.scatter(xs, ys, zs, marker=marker, c=colors, cmap="tab20", linewidth=0)

    for m in ["audio", "image"]:
        # idxs = [idx for idx, datum in enumerate(data) if datum["word-en"] not in WORDS_SEEN]
        idxs = [idx for idx, datum in enumerate(data) if datum["modality"] == m and datum["word-en"] == selected_word]
        xs, ys, zs = 1.01 * embs1[idxs].T
        marker = MODALITY_TO_MARKER[m]
        dark_gray = (0.15, 0.15, 0.15)
        scat = ax.scatter(xs, ys, zs, marker=marker, color=dark_gray, alpha=0.5, linewidth=0)

    light_gray = (0.99, 0.99, 0.99)
    ax.plot_surface(*make_sphere(0.95), color=light_gray, alpha=1.0, linewidth=0)

    ax.axis("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    st.pyplot(fig)