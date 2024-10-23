from pathlib import Path

import pdb
import wn

from nltk.corpus import wordnet as wnnltk
from tqdm import tqdm

import numpy as np
import pandas as pd

from mevgs.utils import read_file, cache_csv

ROOT_THINGS = Path("/mnt/private-share/speechDatabases/THINGS")

path_words = ROOT_THINGS / "02_object-level" / "ids-words_single-tables" / "words.csv"
words = read_file(path_words)

# fmt: off
path_wordnet_id = ROOT_THINGS / "02_object-level" / "ids-words_single-tables" / "wordnet-id.csv"
wordnet_ids = read_file(path_wordnet_id)
# fmt: on

# print(len(words))
# print(len(wordnet_ids))

# num_empty = sum(1 for word in wordnet_ids if word == "")
# print(num_empty)
# print()

# en = wn.Wordnet("oewn:2023")
# pdb.set_trace()

# word, pos, sense = wid.split(".")
# word = word.replace("_", " ")

# if wid == "fish.n.01":
#     pdb.set_trace()

# try:
#     sense = int(sense) - 1
#     synset = en.synsets(word, pos)[sense]
# except:
#     print("ERROR", word, pos, sense)
#     pdb.set_trace()

# print(word)
# print(synset.definition())

# for synset1 in synset.translate(lang="fr"):
#     for word1 in synset1.words():
#         print(word1.lemma())

# print()


def find_missing_wordnet_ids():
    for word, wid in zip(words, wordnet_ids):
        if wid:
            continue

        print(word)
        for s in wnnltk.synsets(word, pos=wnnltk.NOUN):
            print(s.name(), s.definition())
        print()


def get1(i):
    wid = wordnet_ids[i]
    if wid == "":
        return {
            "word": words[i],
            "wordnet-id": "",
            "definition": "",
            "lemma-eng": "",
            "lemma-fra": "",
            "lemma-nld": "",
        }
    else:
        synset = wnnltk.synset(wid)
        return {
            "word": words[i],
            "wordnet-id": wid,
            "definition": synset.definition(),
            "lemma-eng": ", ".join(synset.lemma_names(lang="eng")),
            "lemma-fra": ", ".join(synset.lemma_names(lang="fra")),
            "lemma-nld": ", ".join(synset.lemma_names(lang="nld")),
        }


SHORT_ISO_LANG = {
    "eng": "en",
    "fra": "fr",
    "nld": "nl",
}


def get_counts(df, lang):
    import re
    import pandas as pd

    ROOT = Path(
        "/mnt/private-share/speechDatabases/common-voice-19/cv-corpus-19.0-2024-09-13"
    )

    def load1(split):
        lang_short = SHORT_ISO_LANG[lang]
        path = ROOT / lang_short / f"{split}.tsv"
        with open(path) as f:
            lines = f.readlines()
            sents = [line.split("\t")[3] for line in lines[1:]]
            return sents

    sents = [sent for split in ["train", "dev", "test"] for sent in load1(split)]

    col = f"lemma-{lang}"
    series = df[col]
    idxs = ~series.isna()
    words = series[idxs].str.split(", ").explode().str.replace("_", " ")
    words = words.unique().tolist()
    patterns = {
        word: re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        for word in words
    }

    counts = {word: 0 for word in words}
    for sent in tqdm(sents):
        for _, word in enumerate(words):
            if patterns[word].search(sent):
                counts[word] += 1

    return counts


def prepare_words_things():
    return [get1(i) for i in tqdm(range(len(words)))]


def add_counts(df):
    counts_fr = get_counts(df, "fra")

    def count_total(lemma):
        if pd.isnull(lemma):
            return 0
        else:
            return sum(counts_fr.get(lemma1, 0) for lemma1 in lemma.split(", "))

    def add_counts_to_lemma(lemma):
        if pd.isnull(lemma):
            return lemma
        else:
            return ", ".join(
                f"{lemma1} ({counts_fr.get(lemma1, 0)})" for lemma1 in lemma.split(", ")
            )

    df["counts-fra"] = df["lemma-fra"].map(count_total)
    df["lemma-fra"] = df["lemma-fra"].map(add_counts_to_lemma)
    return df


if __name__ == "__main__":
    find_missing_wordnet_ids()
    # words_things_1 = cache_csv("output/words-things.csv", prepare_words_things)
    # words_things_2 = cache_csv("output/words-things-2.csv", add_counts, words_things_1)
