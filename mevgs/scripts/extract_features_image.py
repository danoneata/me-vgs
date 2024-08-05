import pdb

from functools import partial

import click
import h5py
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
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
    DEVICE = "cuda"

    class MyDataset:
        def __init__(self):
            me_dataset = MEDataset(split, LANGS)
            self.image_files = me_dataset.image_files

        def __getitem__(self, i):
            datum = self.image_files[i]
            return {
                "image": load_image(datum, transform_image_test),
                "name": datum["name"],
            }

        def __len__(self):
            return len(self.image_files)

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    def extract1(image):
        with torch.no_grad():
            image = image.to(DEVICE)
            feature = feature_extractor(image)
            feature = feature.cpu().numpy()
            return feature

    path_hdf5 = f"output/features-image/{DATASET_NAME}-{split}-{feature_type}.h5"

    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=512, num_workers=8)

    with h5py.File(path_hdf5, "a") as f:
        for batch in tqdm(dataloader):
            features = extract1(batch["image"])

            for name, feature in zip(batch["name"], features):
                try:
                    group = f.create_group(name)
                except ValueError:
                    group = f[name]
                    if "feature" in group:
                        continue
                    else:
                        pass
                group.create_dataset("feature", data=feature)



if __name__ == "__main__":
    main()
