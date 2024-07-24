# Solved by using the `forward` method when computing the loss (and avoiding to unwrap the `module`).
# See relevant discussions:
# - https://discuss.pytorch.org/t/loss-and-accuracy-changed-after-saving-and-loading-model/186345
# - https://discuss.pytorch.org/t/is-it-ok-to-use-methods-other-than-forward-in-ddp/176509
# - https://discuss.pytorch.org/t/can-i-access-and-call-submodules-of-ddp-model-when-training/178637
#
from pathlib import Path

import torch

from copy import deepcopy
from tqdm import tqdm

from mevgs.config import CONFIGS
from mevgs.train import setup_model
from mevgs.predict import evaluate_model_ignite


def evaluate_all_checkpoints(rank):
    def load_model(model, config_name, config, i):
        folder = Path("output") / config_name / "checkpoints" # / str(rank)
        checkpoints = sorted(list(folder.iterdir()), reverse=True)
        checkpoint = checkpoints[i]
        state = torch.load(checkpoint)
        print(checkpoint)

        model.load_state_dict(state)
        return model

    config_name = "test"
    config = CONFIGS[config_name]
    config = deepcopy(config)

    device = config["device"]
    feature_type = config["data"]["feature_type"]

    model = setup_model(**config["model"])
    model.eval()
    model.to(config["device"])

    TESTS = [
        "familiar-familiar",
        "novel-familiar",
    ]

    for i in range(3):
        model = load_model(model, config_name, config, i)
        for test_name in TESTS:
            result = evaluate_model_ignite(feature_type, test_name, model, device)
            print(test_name)
            print(result)
            print()


def compare_checkpoints_across_ranks(i):
    config_name = "test"
    folder = Path("output") / config_name / "checkpoints"
    folder0 = folder / "0"
    checkpoints = [path.stem for path in folder0.iterdir()]
    print(checkpoints[i])

    def load_model(rank, i):
        config = CONFIGS[config_name]
        config = deepcopy(config).copy()

        model = setup_model(**config["model"])
        model.eval()
        model.to(config["device"])

        checkpoint = folder / str(rank) / (checkpoints[i] + ".pt")
        state = torch.load(checkpoint)
        model.load_state_dict(state)

        return model

    def load_weights(rank, i):
        checkpoint = folder / str(rank) / (checkpoints[i] + ".pt")
        return torch.load(checkpoint, map_location="cpu")

    w0 = load_weights(0, i)
    w1 = load_weights(1, i)

    def compare_weights(w0, w1):
        for k0, k1 in zip(w0.keys(), w1.keys()):
            assert k0 == k1
            diff = (w0[k0] - w1[k1]).abs().max()
            diff = diff.item()
            if diff > 1e-6:
                print(k0, diff)
                # import pdb; pdb.set_trace()

    compare_weights(w0, w1)


evaluate_all_checkpoints(0)
# evaluate_all_checkpoints(1)
# print()
# compare_checkpoints_across_ranks(0)
# compare_checkpoints_across_ranks(1)
# compare_checkpoints_across_ranks(2)
