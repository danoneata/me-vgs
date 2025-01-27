from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from mevgs.utils import read_file, read_json


sns.set(style="whitegrid", font="Arial", context="talk")


def show_counts():
    FILENAMES = {
        "seen": {
            "audio": ["audio-train-3", "audio-valid-3"],
            "image": ["image-train"],
        },
        "unseen": {
            "audio": ["audio-test-3"],
            "image": ["image-test"],
        },
    }

    GROUPBY_COLS = {
        "audio": ["word-en", "lang"],
        "image": ["word-en"],
    }

    def get_counts(word_type, modality):
        data = [
            datum
            for filename in FILENAMES[word_type][modality]
            for datum in read_json(f"data/filelists/{filename}.json")
        ]

        cols = GROUPBY_COLS[modality]
        df = pd.DataFrame(data)
        df = df.groupby(cols).size().reset_index(name="count")
        return df

    dfs = {
        (word_type, modality): get_counts(word_type, modality)
        for word_type in ("seen", "unseen")
        for modality in ("audio", "image")
    }
    # st.write(dfs[("unseen", "audio")])
    # st.write(dfs[("unseen", "image")])
    fig, axs = plt.subplots(ncols=2, figsize=(9, 7), sharey=True)
    sns.barplot(
        data=dfs[("seen", "audio")],
        x="count",
        y="word-en",
        hue="lang",
        hue_order=["english", "french", "dutch"],
        ax=axs[0],
    )
    sns.barplot(
        data=dfs[("seen", "image")],
        x="count",
        y="word-en",
        width=0.3,
        ax=axs[1],
    )

    f = 10
    for container, number in zip(axs[0].containers, dfs[("seen", "audio")]["count"]):
        axs[0].bar_label(container, labels=[f"{number}"], fontsize=f)
    axs[1].bar_label(axs[1].containers[0], fontsize=f)

    axs[0].set_title("Audio samples")
    axs[1].set_title("Image samples")
    axs[0].set_ylabel("Category")
    axs[0].set_xlabel("Count")
    axs[1].set_xlabel("Count")

    leg = axs[0].get_legend()
    leg.set_title("Language")
    for t in leg.texts:
        t.set_text(t.get_text().capitalize())

    fig.set_tight_layout(True)
    st.pyplot(fig)
    fig.savefig("output/interspeech25/dataset-stats.pdf")


def count_train_and_valid():
    data_train = read_json("data/filelists/audio-train-3.json")
    data_valid = read_json("data/filelists/audio-valid-3.json")
    data = data_train + data_valid

    data = [(datum["word-en"], datum["lang"]) for datum in data]
    counts = Counter(data)

    df = pd.DataFrame(counts.items(), columns=["word-lang", "count"])
    df["word"] = df["word-lang"].apply(lambda x: x[0])
    df["lang"] = df["word-lang"].apply(lambda x: x[1])
    df = df.drop(columns=["word-lang"])
    df = df.pivot(index="word", columns="lang", values="count")
    print(df)

    translations = read_file("data/concepts.csv", lambda x: x.strip().split(","))
    en_to_nl = {en: nl for en, nl, _ in translations}
    en_to_fr = {en: fr for en, _, fr in translations}

    df["nl"] = df.index.map(en_to_nl)
    df["fr"] = df.index.map(en_to_fr)
    df = df.reset_index()
    df = df.rename(columns={"word": "en"})
    column_order = ["en", "english", "fr", "french", "nl", "dutch"]
    print(df[column_order].to_csv(index=False))


def count_train_images():
    data = read_json("data/filelists/image-train.json")
    data = [datum["word-en"] for datum in data]
    words = sorted(set(data))
    counts = Counter(data)
    for word in words:
        print(f"{word},{counts[word]}")


def count_novel():
    words = read_file("data/words-unseen.txt")
    print(words)


def load_words(lang):
    LANG_INDEX = {
        "english": 0,
        "dutch": 1,
        "french": 2,
    }
    i = LANG_INDEX[lang]

    concepts = read_file("data/concepts.csv", lambda x: x.strip().split(","))
    en_to_x = {line[0]: line[i] for line in concepts}

    def translate(words):
        return [en_to_x.get(word, word) for word in words]

    return {
        "seen": translate(read_file("data/words-seen.txt")),
        "unseen": translate(read_file("data/words-unseen.txt")),
    }


def count_in_files(lang):
    words = load_words(lang)

    files = Path(f"data/{lang}_words").iterdir()
    words_in_files = [file.stem.split("_")[0] for file in files]
    counts = Counter(words_in_files)

    words["other"] = sorted(
        list(set(counts.keys()) - set(words["seen"] + words["unseen"]))
    )

    for t in "seen", "unseen", "other":
        for word in words[t]:
            print(f"{word},{counts[word]}")


def compare_counts():
    data1 = {
        split: read_json(f"data/filelists/audio-{split}.json")
        for split in ("train", "valid", "test")
    }
    data2 = {
        split: read_json(f"data/filelists/audio-{split}-3.json")
        for split in ("train", "valid", "test")
    }
    words = {
        split: set(datum["word-en"] for datum in data2[split])
        for split in ("train", "valid", "test")
    }
    for split in ("train", "valid", "test"):
        for lang in ("english", "dutch", "french"):
            for word in words[split]:
                count1 = sum(
                    1
                    for datum in data1[split]
                    if datum["lang"] == lang and datum["word-en"] == word
                )
                count2 = sum(
                    1
                    for datum in data2[split]
                    if datum["lang"] == lang and datum["word-en"] == word
                )
                print(f"{split} {lang} {word}: {count1} → {count2}")


if __name__ == "__main__":
    # count_in_files("dutch")
    # compare_counts()
    # count_train_images()
    # count_train_and_valid()
    show_counts()
