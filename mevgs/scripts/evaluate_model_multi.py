import sys
import pandas as pd
from mevgs.predict import CONFIGS, MODELS, evaluate_model_batched, predict_model_batched

model_name_generic = sys.argv[1]

DEVICE = "cuda"
TESTS = [
    "familiar-familiar",
    "novel-familiar",
]

results = []
for v in "abcde":
    model_name = model_name_generic.format(v)
    model = MODELS[model_name]()
    model.to(DEVICE)

    config = CONFIGS[model_name]
    langs = config["data"]["langs"]

    for lang in langs:
        for test_name in TESTS:
            accuracy = evaluate_model_batched(config, lang, test_name, model, DEVICE)
            result = {
                "variant": v,
                "lang": lang,
                "test_name": test_name,
                "accuracy": accuracy,
            }
            print(result)
            results.append(result)

df = pd.DataFrame(results)
df1 = df.groupby(["lang", "test_name"])["accuracy"].agg(["mean", "std"])
df1["mean-std"] = df1.apply(lambda x: "{:.1f}±{:.1f}".format(x[0], x[1]), axis=1)
print(df1.unstack().to_csv())