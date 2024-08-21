from collections import Counter
from pathlib import Path

import pandas as pd

from mevgs.utils import read_file, read_json


def count_train_and_valid():
    data_train = read_json("data/filelists/audio-train.json")
    data_valid = read_json("data/filelists/audio-valid.json")
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


def count_novel():
    words = read_file("data/words-unseen.txt")
    print(words)


def count_in_files(lang):
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

    words = {
        "seen": translate(read_file("data/words-seen.txt")),
        "unseen": translate(read_file("data/words-unseen.txt")),
    }

    files = Path(f"data/{lang}_words").iterdir()
    words_in_files = [file.stem.split("_")[0] for file in files]
    counts = Counter(words_in_files)

    words["other"] = sorted(list(set(counts.keys()) - set(words["seen"] + words["unseen"])))

    for t in "seen", "unseen", "other":
        for word in words[t]:
            print(f"{word},{counts[word]}")


def compare_counts():
    data1 = {split: read_json(f"data/filelists/audio-{split}.json") for split in ("train", "valid", "test")}
    data2 = {split: read_json(f"data/filelists/audio-{split}-2.json") for split in ("train", "valid", "test")}
    words = {split: set(datum["word-en"] for datum in data2[split]) for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        for lang in ("english", "dutch", "french"):
            for word in words[split]:
                count1 = sum(1 for datum in data1[split] if datum["lang"] == lang and datum["word-en"] == word)
                count2 = sum(1 for datum in data2[split] if datum["lang"] == lang and datum["word-en"] == word)
                print(f"{split} {lang} {word}: {count1} → {count2}")


# count_in_files("dutch")
compare_counts()
