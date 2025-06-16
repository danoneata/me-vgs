# Mutual exclusivity in visually-grounded speech models

This repository represents the implementation of the paper:

> Oneata, Dan, Leanne Nortje, Yevgen Matusevych, and Herman Kamper.
> [The mutual exclusivity bias of bilingual visually grounded speech models.](https://arxiv.org/abs/2506.04037)
> Interspeech, 2025.

## Setup

The implementation relies on PyTorch, which can be installed via conda:

```bash
conda create -n me-vgs python=3.12
conda activate me-vgs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then we can install the code as a library:

```bash
pip install -e .
```

## Example usage

To train a visually grounded speech model, we can run the `mevgs/train.py` script followed by the configuration name;
for example:

```bash
python mevgs/train.py en-nl_links-no_size-md_a 
```

The list of configurations is in the [`mevgs/config.py`](mevgs/config.py) file.

To obtain predictions for the mutual exclusivity tests, we can run the `mevgs/predict.py` script,
again followed by the configuration name;
for example:

```bash
python mevgs/predict.py en-nl_links-no_size-md_a 
```

The results are then obtained with the `mevgs/scripts/evaluate.py` script:

```bash
python mevgs/scripts/evaluate.py en-nl_links-no_size-md_a 
```

The results in the Table 1 from the paper are obtained with the following command:

```bash
python mevgs/scripts/show_interspeech25_table_1.py
```
