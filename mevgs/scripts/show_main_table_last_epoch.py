import pdb
import pandas as pd
from itertools import combinations

from tbparse import SummaryReader
from mevgs.scripts.prepare_predictions_for_yevgen import NAMES


LANG_SHORT_TO_LONG = {
    "en": "english",
    "fr": "french",
    "nl": "dutch",
}


def get_result(train_langs, has_links, model_size, test_lang):
    model_name = "{}_links-{}_size-{}".format(
        "-".join(train_langs),
        has_links,
        model_size,
    )

    test_lang_long = LANG_SHORT_TO_LONG[test_lang]

    def get1(seed):
        folder = NAMES[model_name].format(seed)
        path = f"output/{folder}"
        results = SummaryReader(path)
        scalars = results.scalars
        col1 = "valid/loss"

        # print(scalars["tag"].unique())

        col_nf = f"test/{test_lang_long}/accuracy-novel-familiar"
        if col_nf not in scalars["tag"].unique():
            col_nf = f"test/accuracy-novel-familiar"

        col_ff = f"test/{test_lang_long}/accuracy-familiar-familiar"
        if col_ff not in scalars["tag"].unique():
            col_ff = f"test/accuracy-familiar-familiar"

        df = scalars[scalars["tag"].isin([col1, col_nf, col_ff])]
        df = df.pivot(index="step", columns="tag", values="value")
        best_epoch = df[col1].idxmin()

        return {
            "epoch-best": best_epoch,
            "NF": 100 * df.loc[best_epoch, col_nf],
            "NF-best": 100 * df[col_nf].max(),
            "NF-last": 100 * df.loc[24, col_nf],
            "FF": 100 * df.loc[best_epoch, col_ff],
            "FF-last": 100 * df.loc[24, col_ff],
        }

    df = [{"seed": s, **get1(s)} for s in "abcde"]
    # df = [{"seed": s, **get1(s)} for s in "abcdefghij"]
    df = pd.DataFrame(df)

    col_val = "NF-last"
    nf_value = "{:.1f}±{:.1f}".format(df[col_val].mean(), df[col_val].std())
    # col_val = "FF"
    # ff_value = "{:.1f}±{:.1f}".format(df[col_val].mean(), df[col_val].std())

    print(".", end="")
    
    return {
        "train-langs": train_langs,
        "has-links": has_links,
        "model-size": model_size,
        "test-lang": test_lang,
        "NF": nf_value,
        # "FF": ff_value,
    }


if __name__ == "__main__":
    SIZES = "sm md lg".split()
    TRAIN_LANGS = list(LANG_SHORT_TO_LONG.keys())

    def get_train_langs_combinations(test_lang):
        return [(test_lang, )]  + [
            tuple(sorted([test_lang, other]))
            for other in TRAIN_LANGS
            if other != test_lang
        ]

    results = [
        get_result(train_langs, "no", model_size, test_lang)
        # for train_langs in [("fr", ), ("fr", "nl")]
        for train_langs in [("nl", ), ("en", "nl"), ("fr", "nl")]
        for test_lang in train_langs
        # for test_lang in TRAIN_LANGS
        # for train_langs in get_train_langs_combinations(test_lang)
        for model_size in SIZES
    ]
    cols = [("NF", s) for s in SIZES]
    df = pd.DataFrame(results)
    df = df.set_index(["test-lang", "train-langs", "has-links", "model-size"])
    df = df.unstack("model-size")
    df = df[cols]
    print()
    print(df)
    print(df.to_csv())