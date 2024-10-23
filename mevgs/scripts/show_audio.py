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
    # audios = [audio for audio in audios if len(audio["name"].split("_")) == 2 and audio["name"].split("_")[1].startswith("fn")]
    st.markdown("### {} → {} ◇ num. audios: {}".format(word_en, word, len(audios)))
    if len(audios) > num_samples:
        audios_sample = random.sample(audios, num_samples)
    else:
        audios_sample = audios
    for audio in audios_sample:
        path = get_audio_path(audio)
        st.markdown("`{}`".format(path))
        st.audio(path)
    st.write("---")
