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
            "NF": 100 * df.loc[best_epoch, col_nf],
            "FF": 100 * df.loc[best_epoch, col_ff],
        }

    df = [{"seed": s, **get1(s)} for s in "abcde"]
    df = pd.DataFrame(df)

    print(df)

    nf_mean = df["NF"].mean()
    ff_mean = df["FF"].mean()

    # nf_std = 2 * df["NF"].std()
    # ff_std = 2 * df["FF"].std()

    # template = "{:.1f}({:.1f})"

    nf_std = df["NF"].std()
    ff_std = df["FF"].std()

    template = "{:.1f}±{:.1f}"
   
    return {
        "train-langs": train_langs,
        "has-links": has_links,
        "model-size": model_size,
        "test-lang": test_lang,
        "NF": template.format(nf_mean, nf_std),
        "FF": template.format(ff_mean, ff_std),
    }


if __name__ == "__main__":
    LANGS = ["en", "fr", "nl"]

    def get_train_langs_combinations(test_lang):
        return [(test_lang, )]  + [
            tuple(sorted([test_lang, other]))
            for other in LANGS
            if other != test_lang
        ]

    results = [
        get_result(train_langs, "no", "lg", test_lang)
        for test_lang in LANGS
        for train_langs in get_train_langs_combinations(test_lang)
    ]
    df = pd.DataFrame(results)
    print(df)

    # for s in "sm md lg".split():
    #     print(get_result(("fr", "nl"), "no", s, "fr"))

    # print(get_result(("en", ), "no", "md", "nl"))
    # print(get_result(("en", "nl"), "no", "md", "nl"))
