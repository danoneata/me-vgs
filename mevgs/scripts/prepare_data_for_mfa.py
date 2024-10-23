import os
import pdb
import re

from itertools import groupby
from pathlib import Path
from tqdm import tqdm

from mevgs.scripts.compute_word_statistics import load_words
from mevgs.utils import read_file, cache_json


# fmt: off
DIR_MLS = Path("/mnt/private-share/speechDatabases/multilingual-librispeech/mls_dutch_opus")
DIR_OUT = Path("data/mls-dutch-selected")
SPLITS = ["train", "dev", "test"]
# fmt: on


def read_transcripts(split):
    def parse(line):
        id_, text = line.split("\t")
        speaker_id, chapter_id, *_ = id_.split("_")
        return {
            "id": id_,
            "text": text.strip(),
            "split": split,
            "speaker": speaker_id,
            "chapter": chapter_id,
        }

    path = DIR_MLS / split / "transcripts.txt"
    return read_file(path, parse)


def load_data_selected():
    words = load_words("dutch")
    words = words["seen"] + words["unseen"]
    patterns = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in words]
    data = [datum for split in SPLITS for datum in read_transcripts(split)]
    return [
        datum
        for datum in tqdm(data)
        if any(pattern.search(datum["text"]) for pattern in patterns)
    ]


def prepare_folder():
    data = cache_json("data/mls-dutch-selected.json", load_data_selected)
    data = sorted(data, key=lambda datum: datum["speaker"])

    for speaker_id, group in groupby(data, key=lambda datum: datum["speaker"]):

        folder = DIR_OUT / speaker_id
        os.makedirs(folder, exist_ok=True)

        for datum in group:
            id_ = datum["id"]

            path = folder / (id_ + ".txt")
            with open(path, "w") as f:
                f.write(datum["text"])

            src = (
                DIR_MLS
                / datum["split"]
                / "audio"
                / speaker_id
                / datum["chapter"]
                / (id_ + ".opus")
            )
            dst = folder / (id_ + ".opus")
            os.symlink(src, dst)


def extract_words():
    import textgrid
    import streamlit as st
    import numpy as np
    from pydub import AudioSegment

    st.set_page_config(layout="wide")

    # word = "vogel"
    # # id1 = "2450_5832_008328"
    # # id1 = "2450_7759_005808"
    # id1 = "2450_6613_002413"

    data = [
        "beer_2450_5832_008328",
        "vogel_2450_7759_005808",
        "vogel_2450_6613_002413",
        "boot_2450_5518_000858",
        "auto_1724_11136_000595",
        "kat_2450_8979_001669",
        "klok_2450_4254_000828",
        "klok_2450_7371_000748",
        "koe_2450_5727_003738",
        "koe_2450_3846_005677",
        "koe_1666_1841_002770",
        "hond_2450_3846_003148",
        "hond_2450_4634_000668",
        "olifant_2450_5832_001766",
        "olifant_2450_4029_007837",
        "paard_2450_10646_000169",
    ]

    def do1(word, id1):
        speaker = id1.split("_")[0]
        path = f"data/mls-dutch-selected-aligned-3/{speaker}/{id1}.TextGrid"
        tg = textgrid.TextGrid.fromFile(path)
        intervals = [interval for interval in tg[0] if interval.mark == word]
        interval = intervals[0]
        α, ω = interval.minTime, interval.maxTime

        path_txt = f"data/mls-dutch-selected/{speaker}/{id1}.txt"
        text = read_file(path_txt)
        text = text[0]

        path = f"data/mls-dutch-selected/{speaker}/{id1}.opus"
        audio = AudioSegment.from_file(path, codec="opus", frame_rate=48_000)
        audio = audio[1000 * α: 1000 * ω]
        audio_np = audio.get_array_of_samples()

        st.write(f"{word} ({id1})")
        st.audio(path)
        st.markdown(text)
        col1, col2 = st.columns(2)
        col1.audio(np.array(audio_np), sample_rate=48_000)
        col2.audio(f"data/dutch_words/{word}_{id1}.wav")
        st.markdown("---")

    for datum in data:
        word, *id1 = datum.split("_")
        do1(word, "_".join(id1))


def main():
    extract_words()


if __name__ == "__main__":
    main()
