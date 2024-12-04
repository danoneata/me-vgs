import pandas as pd

from copy import deepcopy

import click

from mevgs.predict import CONFIGS, MODELS, predict_model_batched, load_model_random

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


def load_model_and_config(train_langs, links, size, test_lang, seed):
    train_langs_1 = train_langs if train_langs else ("en", )
    train_langs_1 = "-".join(train_langs_1)

    name = f"{train_langs_1}_links-{links}_size-{size}"
    model_name_generic = NAMES[name]
    model_name = model_name_generic.format(seed)

    config = CONFIGS[model_name]
    config = deepcopy(config)

    if train_langs is not None and test_lang is not None:
        error_message = "Language {} not in model config {}".format(
            test_lang,
            config["data"]["langs"],
        )
        assert test_lang in config["data"]["langs"], error_message

    if train_langs is not None:
        model = MODELS[model_name]()
    else:
        model = load_model_random(None, config)

    model.to(DEVICE)

    return model, config


LANG_SHORT = {"english": "en", "french": "fr", "dutch": "nl"}
LANG_LONG = {v: k for k, v in LANG_SHORT.items()}


def get_preds_1(train_langs, links, size, lang, v):
    lang_long = LANG_LONG[lang]
    model, config = load_model_and_config(train_langs, links, size, lang_long, v)
    return predict_model_batched(config, lang_long, TEST, model, DEVICE)


def merge(ps):
    keys1 = ["audio", "image-pos", "image-neg"]
    keys2 = ["is-correct"]
    n = len(ps)
    dict1 = {k: ps[0][k] for k in keys1}
    dict2 = {"{}/{}".format(k, i): ps[i][k] for i in range(n) for k in keys2}
    return {**dict1, **dict2}

@click.command()
@click.option("--train-langs", "train_langs")
@click.option("--links")
@click.option("--size")
@click.option("--test-lang", "test_lang")
def main(train_langs, links, size, test_lang):
    train_langs = train_langs.split("-")
    preds = [get_preds_1(train_langs, links, size, test_lang, v) for v in "abcde"]
    preds = [merge(ps) for ps in zip(*preds)]

    train_langs_str = "-".join(train_langs)
    name = f"{train_langs_str}_links-{links}_size-{size}"

    df = pd.DataFrame(preds)
    df.to_csv(f"output/preds/{name}_on-{test_lang}.csv", index=False)

    cols = ["is-correct/{}".format(i) for i in range(5)]
    scores = 100 * df[cols].mean()
    print(scores)
    print("{}: {:.2f}±{:.1f}".format(name, scores.mean(), scores.std()))


if __name__ == "__main__":
    main()
