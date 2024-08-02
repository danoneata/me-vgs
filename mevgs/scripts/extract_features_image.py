import pdb

from functools import partial

import click
import h5py
import torch
import torch.nn as nn

from tqdm import tqdm

from mevgs.data import MEDataset, Split, load_image, transform_image_test


class ImageBackboneDINO(nn.Module):
    def __init__(self, type_):
        assert type_ == "resnet50"
        super(ImageBackboneDINO, self).__init__()

        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_" + type_,
            pretrained=True,
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        return self.model(x)


FEATURE_EXTRACTORS = {
    # "alexnet": ImageBackboneAlexNet,
    "dino-resnet50": partial(ImageBackboneDINO, type_="resnet50"),
}


SPLITS = "train valid test".split()
@click.command()
@click.option("-s", "--split", type=click.Choice(SPLITS), required=True)
@click.option("-f", "--feature-type", type=str, required=True)
def main(split: Split, feature_type: str):
    DATASET_NAME = "me-dataset"
    LANGS = ("english",)
    dataset = MEDataset(split, LANGS)
    dataset = dataset.image_files

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    def extract1(image):
        with torch.no_grad():
            feature = feature_extractor(image)
        feature = feature.squeeze(dim=0)
        feature = feature.cpu().numpy()
        return feature

    langs = "_".join(LANGS)
    path_hdf5 = f"output/features-image/{DATASET_NAME}-{split}-{langs}-{feature_type}.h5"

    with h5py.File(path_hdf5, "a") as f:
        for i in tqdm(range(len(dataset))):
            datum = dataset[i]
            try:
                group = f.create_group(datum["name"])
            except ValueError:
                group = f[datum["name"]]
                if "feature" in group:
                    continue
                else:
                    pass

            image = load_image(datum, transform_image_test)
            feature = extract1(image)
            group.create_dataset("feature", data=feature)


if __name__ == "__main__":
    main()
