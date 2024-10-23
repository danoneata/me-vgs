import sys
import pandas as pd

from copy import deepcopy

from mevgs.predict import CONFIGS, MODELS, predict_model_batched

DEVICE = "cuda"
TEST = "novel-familiar"


NAMES = {
    # row 3
    "en_links-no_size-sm": "26{}-sm-en",
    "en_links-no_size-md": "26{}",
    "en_links-no_size-lg": "26{}-lg-en",
    # row 4
    "en-fr_links-no_size-sm": "26{}-sm-en-fr",
    "en-fr_links-no_size-md": "26{}-en-fr",
    "en-fr_links-no_size-lg": "29{}-en-fr",
    # row 5
    "en_links-yes_size-sm": "clip-lang-links-{}-sm-en",
    "en_links-yes_size-md": "clip-lang-links-{}-md-en",
    "en_links-yes_size-lg": "clip-lang-links-{}-lg-en",
    # row 6
    "en-fr_links-yes_size-sm": "clip-lang-links-{}-sm-en-fr",
    "en-fr_links-yes_size-md": "clip-lang-links-{}-md-en-fr",
    "en-fr_links-yes_size-lg": "clip-lang-links-{}-lg-en-fr",
    # row 9
    "fr_links-no_size-sm": "26{}-sm-fr",
    "fr_links-no_size-md": "26{}-fr",
    "fr_links-no_size-lg": "26{}-lg-fr",
    # row 11
    "fr_links-yes_size-sm": "clip-lang-links-{}-sm-fr",
    "fr_links-yes_size-md": "clip-lang-links-{}-md-fr",
    "fr_links-yes_size-lg": "clip-lang-links-{}-lg-fr",
}

for lang1 in "nl en-nl fr-nl".split():
    for size1 in "sm md lg".split():
        key = f"{lang1}_links-no_size-{size1}"
        val = key + "_{}"
        NAMES[key] = val


def load_model_and_config(name, lang, v):
    model_name_generic = NAMES[name]
    model_name = model_name_generic.format(v)

    config = CONFIGS[model_name]
    config = deepcopy(config)
    assert lang in config["data"]["langs"], "Language {} not in model config {}".format(lang, config["data"]["langs"])

    model = MODELS[model_name]()
    model.to(DEVICE)

    return model, config


def get_preds_1(name, lang, v):
    model, config = load_model_and_config(name, lang, v)
    return predict_model_batched(config, lang, TEST, model, DEVICE)


def merge(ps):
    keys1 = ["audio", "image-pos", "image-neg"]
    keys2 = ["is-correct"]
    n = len(ps)
    dict1 = {k: ps[0][k] for k in keys1}
    dict2 = {"{}/{}".format(k, i): ps[i][k] for i in range(n) for k in keys2}
    return {**dict1, **dict2}


if __name__ == "__main__":
    name = sys.argv[1]
    lang = sys.argv[2]

    preds = [get_preds_1(name, lang, v) for v in "abcde"]
    preds = [merge(ps) for ps in zip(*preds)]

    LANG_SHORT = {"english": "en", "french": "fr", "dutch": "nl"}
    lang_short = LANG_SHORT[lang]
    df = pd.DataFrame(preds)
    df.to_csv(f"output/preds/{name}_on-{lang_short}.csv", index=False)
    # import pdb; pdb.set_trace()

    cols = ["is-correct/{}".format(i) for i in range(5)]
    scores = 100 * df[cols].mean()
    print("{}: {:.2f}±{:.1f}".format(name, scores.mean(), scores.std()))
