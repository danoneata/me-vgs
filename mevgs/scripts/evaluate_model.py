import sys
from mevgs.predict import CONFIGS, MODELS, evaluate_model_batched, evaluate_model, evaluate_model_ignite

model_name = sys.argv[1]

DEVICE = "cuda"
model = MODELS[model_name]()
model.to(DEVICE)

config = CONFIGS[model_name]
langs = config["data"]["langs"]
# lang_str = langs[0]
# assert len(langs) == 1

# TESTS = [
#     "familiar-familiar",
#     "novel-familiar",
#     "leanne-familiar-familiar",
#     "leanne-novel-familiar",
#     "leanne-1000-familiar-familiar",
#     "leanne-1000-novel-familiar",
# ]

TESTS = [
    "familiar-familiar",
    "novel-familiar",
]

# from ignite.utils import manual_seed
# manual_seed(config["seed"])

results = []
for lang in langs:
    for test_name in TESTS:
        # accuracy = evaluate_model(config, test_name, model, DEVICE)
        accuracy = evaluate_model_batched(config, lang, test_name, model, DEVICE)
        # accuracy = evaluate_model_ignite(config, test_name, model, DEVICE)

        print(test_name)
        print(accuracy)
        results.append(accuracy)

print(",".join(map(str, results)))
