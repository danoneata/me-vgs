from copy import deepcopy
from itertools import groupby
from pathlib import Path
from typing import List, Dict

import json
import pdb
import random
import numpy as np

from toolz import first
from sklearn.model_selection import train_test_split

from mevgs.data import load_dictionary, MEDataset, PairedMEDataset, SimplePairedMEDataset
from mevgs.utils import read_json, read_file


SEED = 42
random.seed(SEED)


DATA_DIR = Path("./data")
WORD_DICT = load_dictionary()


def load_data(split):
    SPLITS = {
        "train": "train",
        "valid": "val",
    }
    split_short = SPLITS[split]

    data = np.load(DATA_DIR / f"{split_short}_lookup.npz", allow_pickle=True)
    return data["lookup"].item()


def save_data(data, modality, split):
    data = deduplicate(data)
    path_out = f"data/filelists/{modality}-{split}.json"
    with open(path_out, "w") as f:
        json.dump(data, f, indent=2)


def deduplicate(data: List[Dict]) -> List[Dict]:
    keys = data[0].keys()
    data1 = [tuple(datum.values()) for datum in data]
    data1 = set(data1)
    return [dict(zip(keys, datum)) for datum in data1]


def extract_word_en(lang, audio_name: str) -> str:
    word = audio_name.split("_")[0]
    entry = first(e for e in WORD_DICT if e[lang] == word)
    return entry["english"]


def prepare_audio_filelist(split):
    data = load_data(split)
    data_audio = [
        {
            "lang": lang,
            "name": audio_name.stem,
            "word-en": extract_word_en(lang, audio_name.stem),
        }
        for _, data1 in data.items()
        for lang, data2 in data1.items()
        if lang != "images"
        for _, audio_names in data2.items()
        for audio_name in audio_names
    ]
    save_data(data_audio, "audio", split)


def prepare_image_filelist(split):
    data = load_data(split)
    data_image = [
        {
            "name": image_name.stem,
            "word-en": image_name.stem.split("_")[0],
        }
        for _, data1 in data.items()
        for lang, data2 in data1.items()
        if lang == "images"
        for _, image_names in data2.items()
        for image_name in image_names
    ]
    save_data(data_image, "image", split)


def prepare_audio_filelists_2():
    """Prepares audio filelists to have Dutch and French in the test set:

    - Resplits original audio filelists to have familiar audios for Dutch and French in the test set.
    - Adds novel familiar audio for Dutch and French in the test set.

    """
    SPLITS = ("train", "valid", "test")

    def are_disjoint(data1, data2):
        data1 = set(datum.values() for datum in data1)
        data2 = set(datum.values() for datum in data2)
        return data1.isdisjoint(data2)

    def move_dutch_french_to_test(data):
        key = lambda datum: (datum["lang"], datum["word-en"])

        data_train = sorted(data["train"], key=key)
        data_out = {
            "train": [],
            "valid": deepcopy(data["valid"]),
            "test": deepcopy(data["test"]),
        }

        for key, group in groupby(data_train, key=key):
            lang, _ = key
            group = list(group)
            if lang in ("dutch", "french"):
                group_tr, group_te = train_test_split(
                    group, test_size=10, random_state=SEED
                )
                data_out["train"].extend(group_tr)
                data_out["test"].extend(group_te)
            elif lang == "english":
                data_out["train"].extend(group)
            else:
                assert False, "Unknown language"

        diff_tr = len(data["train"]) - len(data_out["train"])
        diff_te = len(data_out["test"]) - len(data["test"])
        assert diff_tr == diff_te

        assert are_disjoint(data_out["train"], data_out["test"])
        assert are_disjoint(data_out["train"], data_out["valid"])
        assert are_disjoint(data_out["valid"], data_out["test"])

        return data_out

    def add_novel_familiar(data):
        path = "data/{}_words"
        words_unseen = read_file("data/words-unseen.txt")
        xx_to_en = {
            lang: {
                WORD_DICT[i][lang]: WORD_DICT[i]["english"]
                for i in range(len(WORD_DICT))
            }
            for lang in ("dutch", "french")
        }
        def get_word_en(lang, file):
            word, *_ = file.stem.split("_")
            try:
                return xx_to_en[lang][word]
            except KeyError:
                return None

        data_new = [
            {
                "lang": lang,
                "name": file.stem,
                "word-en": get_word_en(lang, file),
            }
            for lang in ("dutch", "french")
            for file in Path(path.format(lang)).iterdir()
            if get_word_en(lang, file) in words_unseen
        ]
        data_out = {k: deepcopy(v) for k, v in data.items()}
        data_out["test"].extend(data_new)
        return data_out

    path = "data/filelists/audio-{}.json"
    data = {split: read_json(path.format(split)) for split in SPLITS}
    data = move_dutch_french_to_test(data)
    data = add_novel_familiar(data)

    for split in data:
        save_data(data[split], "audio", split + "-2")


def prepare_test_filelist():
    def unpack_datum(datum):
        word, image_path, audio_path, image_source = datum
        entry_image = {
            "name": image_path.stem,
            "word-en": word,
            "source": image_source,
        }
        entry_audio = {
            "lang": "english",
            "name": audio_path.stem,
            "word-en": word,
        }
        return entry_audio, entry_image

    path = "mme/data/episodes.npz"
    data = np.load(path, allow_pickle=True)["episodes"].item()

    data_out = [
        unpack_datum(datum4)
        for datum1 in data.values()
        for datum2 in datum1.values()
        for datum3 in datum2.values()
        for datum4 in datum3
    ]

    data_audio, data_image = zip(*data_out)
    data_audio = deduplicate(data_audio)
    data_image = deduplicate(data_image)

    # Sanity check (seems good!)

    # def extract_names(data):
    #     data1 = data.item().values()
    #     return [datum2.stem for datum1 in data1 for datum2 in datum1]

    # path = "mme/results/files/episode_data.npz"
    # data = np.load(path, allow_pickle=True)

    # audio_names = extract_names(data["audio_1"]) + extract_names(data["audio_2"])
    # audio_names = set(audio_names)
    # image_names = extract_names(data["image_1"]) + extract_names(data["image_2"])
    # image_names = set(image_names)

    # print(len(audio_names), len(image_names))
    # print(len(data_audio), len(data_image))

    # assert audio_names == set([d["name"] for d in data_audio])
    # assert image_names == set([d["name"] for d in data_image])
    # pdb.set_trace()

    save_data(data_audio, "audio", "test")
    save_data(data_image, "image", "test")


def prepare_familiar_familiar(lang, num_word_repeat, split):
    # split = "test"
    assert split in ("train", "valid", "test")
    langs = (lang,)
    dataset = MEDataset(split, langs)

    def sample_neg_train_valid(data, image_pos):
        word = image_pos["word-en"]
        words = set(dataset.words_seen) - set([word])
        word_other = random.choice(list(words))
        return random.choice(data[word_other])

    def sample_neg_test(data, image_pos):
        data1 = [
            datum
            for word in dataset.words_seen
            if word != image_pos["word-en"]
            for datum in data[word]
            if datum["source"] == image_pos["source"]
        ]
        return random.choice(data1)

    SAMPLE_NEGS = {
        "train": sample_neg_train_valid,
        "valid": sample_neg_train_valid,
        "test": sample_neg_test,
    }
    sample_neg = SAMPLE_NEGS[split]

    def sample_pair(word):
        audio = random.choice(dataset.word_to_audios[word])
        image_pos = random.choice(dataset.word_to_images[word])
        image_neg = sample_neg(dataset.word_to_images, image_pos)
        return {
            "audio": audio,
            "image-pos": image_pos,
            "image-neg": image_neg,
        }

    data = [
        sample_pair(word) for word in dataset.words_seen for _ in range(num_word_repeat)
    ]

    with open(f"data/filelists/familiar-familiar-{lang}-{split}.json", "w") as f:
        json.dump(data, f, indent=2)


def prepare_novel_familiar(lang, num_word_repeat, exclude_words):
    split = "test"
    langs = (lang,)
    dataset = MEDataset(split, langs)

    def sample_neg(data, image_pos):
        data1 = [
            datum
            for word in dataset.words_seen
            for datum in data[word]
            if datum["source"] == image_pos["source"]
        ]
        datum = random.choice(data1)
        assert datum["word-en"] != image_pos["word-en"]
        return datum

    def sample_pair(word):
        try:
            audio = random.choice(dataset.word_to_audios[word])
        except:
            pdb.set_trace()
        image_pos = random.choice(dataset.word_to_images[word])
        image_neg = sample_neg(dataset.word_to_images, image_pos)
        return {
            "audio": audio,
            "image-pos": image_pos,
            "image-neg": image_neg,
        }

    data = [
        sample_pair(word)
        for word in dataset.words_unseen
        if word not in exclude_words
        for _ in range(num_word_repeat)
    ]

    with open(f"data/filelists/novel-familiar-{lang}-{split}.json", "w") as f:
        json.dump(data, f, indent=2)


def extract_filelists_from_leanne(type_):
    assert type_ in {"subsample", "full"}

    dataset = MEDataset("test", langs=("english",))

    path = "data/episode_data.npz"
    data = np.load(path, allow_pickle=True)

    audio = data["audio_1"].item()
    audio_words = data["audio_labels_1"].item()

    image_pos = data["image_1"].item()
    image_pos_words = data["image_labels_1"].item()

    image_neg = data["image_2"].item()
    image_neg_words = data["image_labels_2"].item()

    episodes = list(range(1000))

    if type_ == "subsample":
        episodes = random.sample(episodes, 30)

    data = [
        {
            "audio": {
                "name": audio[e][i].stem,
                "word-en": audio_words[e][i],
                "lang": "english",
            },
            "image-pos": {
                "name": image_pos[e][i].stem,
                "word-en": image_pos_words[e][i],
            },
            "image-neg": {
                "name": image_neg[e][i].stem,
                "word-en": image_neg_words[e][i],
            },
        }
        for e in episodes
        for i in range(33)
    ]

    data_ff = [
        datum
        for datum in data
        if datum["audio"]["word-en"] in dataset.words_seen
        and datum["image-pos"]["word-en"] in dataset.words_seen
        and datum["image-neg"]["word-en"] in dataset.words_seen
    ]

    data_nf = [
        datum
        for datum in data
        if datum["audio"]["word-en"] in dataset.words_unseen
        and datum["image-pos"]["word-en"] in dataset.words_unseen
        and datum["image-neg"]["word-en"] in dataset.words_seen
    ]

    SUFFIXES = {
        "subsample": "",
        "full": "-1000",
    }
    suffix = SUFFIXES[type_]

    with open(f"data/filelists/leanne{suffix}-familiar-familiar-test.json", "w") as f:
        json.dump(data_ff, f, indent=2)

    with open(f"data/filelists/leanne{suffix}-novel-familiar-test.json", "w") as f:
        json.dump(data_nf, f, indent=2)


def prepare_validation_samples(langs, num_pos, num_neg):
    # Fix the validation samples to ensure comparable results across runs.
    feature_type_audio = "wavlm-base-plus"
    feature_type_image = "dino-resnet50"
    dataset = PairedMEDataset(
        "valid",
        langs,
        num_pos,
        num_neg,
        feature_type_audio,
        feature_type_image,
        to_fix_validation_samples=False,
    )
    samples = [dataset.get_positives_and_negatives(i) for i in range(len(dataset))]
    suffix = "{}-P{}-N{}".format("_".join(langs), num_pos, num_neg)
    with open(f"data/filelists/validation-samples-{suffix}.json", "w") as f:
        json.dump(samples, f, indent=2)


def prepare_validation_samples_simple(langs):
    # Fix the validation samples to ensure comparable results across runs.
    feature_type_audio = "wavlm-base-plus"
    feature_type_image = "dino-resnet50"
    dataset = SimplePairedMEDataset(
        "valid",
        langs,
        feature_type_audio,
        feature_type_image,
        to_fix_validation_samples=False,
    )
    samples = [dataset.get_positive_pair(i) for i in range(len(dataset))]
    suffix = "_".join(langs)
    with open(f"data/filelists/validation-samples-simple-{suffix}.json", "w") as f:
        json.dump(samples, f, indent=2)


def prepare_audio_filelists_3():
    """Updates the Dutch audio filelists to account for the realignment of the
    MLS dataset. Given that not all audio files were used in the MLS dataset,
    we replace the missing audio with other audio files corresponding to the
    same word.

    """
    import os
    SPLITS = ("train", "valid", "test")
    path = "data/filelists/audio-{}-2.json"
    data = {split: read_json(path.format(split)) for split in SPLITS}
    names_dutch = [datum["name"] for split in SPLITS for datum in data[split] if datum["lang"] == "dutch"]
    names_dutch_files = [f.split(".")[0] for f in os.listdir("data/dutch_words")]
    names_dutch_files = set(names_dutch_files)

    unused = names_dutch_files - set(names_dutch)
    unused = list(unused)
    print(len(unused))

    def pick_unused(name):
        word, *_ = name.split("_")
        indices = [i for i, n in enumerate(unused) if n.split("_")[0] == word]
        try:
            index = random.choice(indices)
            return unused.pop(index)
        except IndexError:
            return None

    def get_datum_new(datum):
        if datum["lang"] == "dutch" and datum["name"] not in names_dutch_files:
            name_old = datum["name"]
            name_new = pick_unused(name_old)
            if name_new is not None:
                return {**datum, "name": name_new}
            else:
                return None
        else:
            return datum

    data2 = {split: [get_datum_new(datum) for datum in data[split]] for split in SPLITS}
    data2 = {split: [datum for datum in data2[split] if datum is not None] for split in SPLITS}

    for split in data:
        save_data(data2[split], "audio", split + "-3")


if __name__ == "__main__":
    # prepare_audio_filelist("train")
    # prepare_audio_filelist("valid")
    # prepare_image_filelist("train")
    # prepare_image_filelist("valid")
    # prepare_familiar_familiar("test", 10)
    # prepare_novel_familiar(10)
    # extract_filelists_from_leanne(type_="full")
    # prepare_validation_samples(("english", ), 1, 11)
    # prepare_validation_samples(("dutch",), 1, 11)
    # prepare_validation_samples(("french",), 1, 11)
    # prepare_validation_samples(("english", "french"), 1, 11)
    # prepare_validation_samples_simple(("english", ))
    # prepare_validation_samples_simple(("french", ))
    # prepare_validation_samples_simple(("english", "french"))
    # prepare_audio_filelists_2()
    # for lang in ("english", "french", "dutch"):
    #     print(lang)
    #     prepare_familiar_familiar(lang, 50, split="test")
    #     prepare_novel_familiar(lang, 50, exclude_words=["nautilus"])

    # 2024-10-10
    prepare_audio_filelists_3()
    prepare_validation_samples(("dutch",), 1, 11)
    prepare_validation_samples_simple(("dutch", ))
    prepare_familiar_familiar("dutch", 50, split="test")
    prepare_novel_familiar("dutch", 50, exclude_words=["nautilus"])
