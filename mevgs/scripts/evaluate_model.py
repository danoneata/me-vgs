import sys
from mevgs.predict import CONFIGS, MODELS, evaluate_model_batched, evaluate_model, evaluate_model_ignite

model_name = sys.argv[1]

DEVICE = "cuda"
model = MODELS[model_name]()
model.to(DEVICE)

config = CONFIGS[model_name]
feature_type = config["data"]["feature_type"]

TESTS = [
    "familiar-familiar",
    "novel-familiar",
    "leanne-familiar-familiar",
    "leanne-novel-familiar",
]

# from ignite.utils import manual_seed
# manual_seed(config["seed"])

results = []
for test_name in TESTS:
    print(test_name)
    # accuracy = evaluate_model(test_name, model, DEVICE)
    # accuracy = evaluate_model_batched(feature_type, test_name, model, DEVICE)
    accuracy = evaluate_model_ignite(feature_type, test_name, model, DEVICE)

    print(accuracy)
    results.append(accuracy)

print(",".join(map(str, results)))
