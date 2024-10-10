# NOTE
# Use the mms-fa environment to run this script
# Based on this tutorial: https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html
# python mevgs/scripts/mms_align.py

import re
import librosa

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st

from tqdm import tqdm

import torch
import torchaudio

from torchaudio.pipelines import MMS_FA as bundle
from pydub import AudioSegment

from mevgs.scripts.compute_word_statistics import load_words
from mevgs.utils import cache_json, read_json


print(torch.__version__)
print(torchaudio.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

words = load_words("dutch")
vocab = set(words["seen"] + words["unseen"])


def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans


def _score(spans):
    # Compute average score weighted by the span length
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def plot_alignments(
    waveform,
    token_spans,
    emission,
    transcript,
    sample_rate=bundle.sample_rate,
):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(
            t0 - 0.5,
            t1 - 0.5,
            facecolor="None",
            hatch="/",
            edgecolor="white",
        )
        axes[1].axvspan(
            ratio * t0,
            ratio * t1,
            facecolor="None",
            hatch="/",
            edgecolor="white",
        )
        axes[1].annotate(
            f"{_score(t_spans):.2f}",
            (ratio * t0, sample_rate * 0.51),
            annotation_clip=False,
        )

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()
    return fig


def preview_word(
    waveform,
    spans,
    num_frames,
    transcript,
    sample_rate=bundle.sample_rate,
):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    segment = waveform[:, x0:x1]
    st.markdown(
        "{} ({}): {} - {} sec".format(
            transcript,
            _score(spans),
            x0 / sample_rate,
            x1 / sample_rate,
        )
    )
    return st.audio(segment.numpy(), sample_rate=sample_rate)


def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


ROOT = Path("data/mls-dutch-selected")


def align1(datum, to_show=False):
    path_audio = ROOT / datum["speaker"] / (datum["id"] + ".opus")
    waveform, sample_rate = librosa.load(path_audio, sr=bundle.sample_rate)

    text_raw = datum["text"]
    text_normalized = normalize_uroman(text_raw)

    if to_show:
        st.markdown("Raw transcript: " + text_raw)
        st.markdown("Normalized transcript: " + text_normalized)
        st.audio(waveform, sample_rate=sample_rate)

    waveform = torch.tensor(waveform).unsqueeze(0)
    transcript = text_normalized.split()
    emission, token_spans = compute_alignments(waveform, transcript)
    num_frames = emission.size(1)

    if to_show:
        fig = plot_alignments(waveform, token_spans, emission, transcript)
        st.pyplot(fig)

    if to_show:
        for j, word in enumerate(transcript):
            if word in vocab:
                preview_word(waveform, token_spans[j], num_frames, transcript[j])

    def to_time(t):
        return t * waveform.size(1) / sample_rate / num_frames

    def to_dict(token, i):
        return {
            "i": i,
            "word": transcript[i],
            "start": to_time(token.start),
            "end": to_time(token.end),
            "score": token.score,
            "token": token.token,
        }

    return [
        [to_dict(token, i) for token in token_span]
        for i, token_span in enumerate(token_spans)
    ]


def extract_words(datum, alignment, to_show):
    DIR_OUT = Path("data/dutch_words_2")
    sr = bundle.sample_rate
    for a in alignment:
        if a[0]["word"] in vocab:
            score_mean = sum(t["score"] for t in a) / len(a)
            if score_mean < 0.3:
                continue

            path_audio = ROOT / datum["speaker"] / (datum["id"] + ".opus")
            audio, _ = librosa.load(path_audio, sr=sr)

            α = int(sr * a[0]["start"])
            ω = int(sr * a[-1]["end"])

            path = DIR_OUT / (a[0]["word"] + "_" + datum["id"] + ".wav")
            audio = audio[α:ω]
            sf.write(path, audio, sr)

            if to_show:
                st.markdown("word: `{}` · score: {:.3f}".format(a[0]["word"], score_mean))
                st.audio(str(path))
                st.write(a)
                st.markdown("---")


to_show = False
data = read_json("data/mls-dutch-selected.json")

for datum in data:
# for datum in tqdm(data):
    # alignment = align1(datum, to_show)
    path_alignment = "data/mls-dutch-selected-aligned-mms/{}.json".format(datum["id"])
    alignment = cache_json(path_alignment, align1, datum, to_show)
    extract_words(datum, alignment, to_show)
    # import pdb; pdb.set_trace()
