import random
import streamlit as st
from mevgs.data import MEDataset, get_audio_path, load_dictionary


with st.sidebar:
    lang = st.selectbox("Language", ["english", "french", "dutch"], index=2)
    split = st.selectbox("Split", ["train", "valid", "test"])
    num_samples = st.number_input("Num. audio samples", 1, 100, 10)


WORD_DICT = load_dictionary()
EN_TO_XX = {
    lang: {WORD_DICT[i]["english"]: WORD_DICT[i][lang] for i in range(len(WORD_DICT))}
    for lang in ("english", "dutch", "french")
}

dataset = MEDataset(split, (lang,))
for word_en, audios in dataset.word_to_audios.items():
    word = EN_TO_XX[lang][word_en]
    st.markdown("### {} → {} ◇ num. audios: {}".format(word_en, word, len(audios)))
    audios_sample = random.sample(audios, num_samples)
    for audio in audios_sample:
        path = get_audio_path(audio)
        st.markdown("`{}`".format(path))
        st.audio(path)
    st.write("---")
