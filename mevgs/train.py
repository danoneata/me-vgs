from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from torch import optim
from torch.nn import functional as F

import click

import ignite.distributed as idist
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, create_lr_scheduler_with_warmup
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import convert_tensor, manual_seed

from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)

from mevgs.config import CONFIGS
from mevgs.data import (
    collate_nested,
    collate_nested_two_audios,
    collate_with_audio,
    PairedMEDataset,
    PairedTestDataset,
    SimplePairedMEDataset,
    TripletMEDataset,
)
from mevgs.model import setup_model


def identity_loss(loss, _):
    return loss


# def info_nce_cross_entropy_loss(pred, true):
# def loss_dir(pred):
#     # pred1 = torch.concat((pred[0, :1], pred[0, true == 0]))
#     # assert true.sum() == 1
#     # pred1 = pred[0]
#     # pred1 = F.softmax(pred1, dim=0)
#     # loss = -pred1[0].log()
#     # return loss
#     pred1 = pred[true == 1]
#     pred1 = F.softmax(pred1, dim=1)
#     pred1 = pred1[:, true == 1].sum(dim=1)
#     loss = -torch.log(pred1).mean()
#     return loss

# loss1 = loss_dir(pred)
# loss2 = loss_dir(pred.T)
# return (loss1 + loss2) / 2

# def loss_dir(pred):
#     pred1 = pred[true == 1]
#     loss1 = F.mse_loss(pred1[:, true == 1], torch.tensor(100.0).to("cuda"))
#     loss2 = F.mse_loss(pred1[:, true == 0], torch.tensor(0.0).to("cuda"))
#     return (loss1 + loss2) / 2

# loss1 = loss_dir(pred)
# loss2 = loss_dir(pred.T)
# return (loss1 + loss2) / 2

# loss1 = F.cross_entropy(pred, true)
# loss2 = F.cross_entropy(pred.T, true)
# return (loss1 + loss2) / 2


TEST_NAMES = ["familiar-familiar", "novel-familiar"]


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


class UtilsDefaultTraining:
    @staticmethod
    def prepare_batch_fn(batch, device, non_blocking):
        batch = {k: convert_tensor(v, device, non_blocking) for k, v in batch.items()}
        inp = batch["audio"], batch["audio-length"], batch["image"], batch["label"]
        out = batch["label"]
        return inp, out

    # @staticmethod
    # def set_bn_eval(m):
    #     classname = m.__class__.__name__
    #     if classname.find("BatchNorm") != -1:
    #         m.eval()

    @staticmethod
    def model_fn(model, inputs):
        # model.apply(UtilsTraining.set_bn_eval)
        return model(*inputs)

    @staticmethod
    def get_metrics(device):
        return {"loss": Loss(identity_loss, device=device)}


class UtilsTwoAudiosTraining(UtilsDefaultTraining):
    @staticmethod
    def prepare_batch_fn(batch, device, non_blocking):
        batch = {k: convert_tensor(v, device, non_blocking) for k, v in batch.items()}
        inp = (
            batch["audio1"],
            batch["audio1-length"],
            batch["audio2"],
            batch["audio2-length"],
            batch["image"],
            batch["label"],
        )
        out = batch["label"]
        return inp, out


class UtilsPairedTest:
    @staticmethod
    def prepare_batch_fn(batch, device, non_blocking):
        B, *_ = batch["image-pos"].shape
        # batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
        batch = {k: convert_tensor(v, device, non_blocking) for k, v in batch.items()}
        inp = (
            batch["audio"],
            batch["audio-length"],
            batch["image-pos"],
            batch["image-neg"],
        )
        out = torch.zeros(B, device=device, dtype=torch.long)
        return inp, out

    @staticmethod
    def output_transform(x, y, y_pred):
        y_pred = y_pred.argmax(dim=1)
        return y_pred, y

    @staticmethod
    def model_fn(model, inputs):
        return unwrap_model(model).predict_paired_test(*inputs)

    @staticmethod
    def get_metrics(lang, test_name, device):
        return {f"{lang}/accuracy-{test_name}": Accuracy(device=device)}


def setup_data_nested(*, num_workers, batch_size, **kwargs_ds):
    train_dataset = PairedMEDataset(split="train", **kwargs_ds)
    valid_dataset = PairedMEDataset(split="valid", **kwargs_ds)

    kwargs_dl = {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "collate_fn": collate_nested,
    }

    train_dataloader = idist.auto_dataloader(train_dataset, **kwargs_dl, shuffle=True)
    valid_dataloader = idist.auto_dataloader(valid_dataset, **kwargs_dl)

    return train_dataloader, valid_dataloader


def setup_data_nested_two_audios(*, num_workers, batch_size, **kwargs_ds):
    train_dataset = TripletMEDataset(split="train", **kwargs_ds)
    valid_dataset = TripletMEDataset(split="valid", **kwargs_ds)

    kwargs_dl = {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "collate_fn": collate_nested_two_audios,
    }

    train_dataloader = idist.auto_dataloader(train_dataset, **kwargs_dl, shuffle=True)
    valid_dataloader = idist.auto_dataloader(valid_dataset, **kwargs_dl)

    return train_dataloader, valid_dataloader


def setup_data_simple(*, num_workers, batch_size, **kwargs_ds):
    train_dataset = SimplePairedMEDataset(split="train", **kwargs_ds)
    valid_dataset = SimplePairedMEDataset(split="valid", **kwargs_ds)

    kwargs_dl = {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "collate_fn": collate_with_audio,
    }

    train_dataloader = idist.auto_dataloader(train_dataset, **kwargs_dl, shuffle=True)
    valid_dataloader = idist.auto_dataloader(valid_dataset, **kwargs_dl)

    return train_dataloader, valid_dataloader


def setup_data_paired_test(
    *, num_workers, batch_size, langs, feature_type_audio, feature_type_image
):
    return {
        lang: {
            test_name: idist.auto_dataloader(
                PairedTestDataset(
                    (lang,),
                    feature_type_audio,
                    feature_type_image,
                    test_name,
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_with_audio,
            )
            for test_name in TEST_NAMES
        }
        for lang in langs
    }


SETUP_DATA_FUNCS = {
    "clip": setup_data_nested,
    "mattnet": setup_data_nested,
    "barlip": setup_data_simple,
    "clip-two-audios": setup_data_nested_two_audios,
}

UTILS_TRAINING_CLASSES = {
    "clip": UtilsDefaultTraining,
    "mattnet": UtilsDefaultTraining,
    "barlip": UtilsDefaultTraining,
    "clip-two-audios": UtilsTwoAudiosTraining,
}


def train(local_rank, config_name: str):
    rank = idist.get_rank()
    config = CONFIGS[config_name]
    manual_seed(config["seed"])

    # Setup output directory
    output_dir = Path(f"output/{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # config.output_dir = output_dir

    setup_data = SETUP_DATA_FUNCS[config["model"]["model_name"]]
    UtilsTraining = UTILS_TRAINING_CLASSES[config["model"]["model_name"]]

    world_size = idist.get_world_size()
    dataloader_train, dataloader_valid = setup_data(**config["data"])
    dataloaders = setup_data_paired_test(
        batch_size=world_size * 16,
        num_workers=4,
        langs=config["data"]["langs"],
        feature_type_audio=config["data"]["feature_type_audio"],
        feature_type_image=config["data"]["feature_type_image"],
    )

    device = config["device"]
    model = setup_model(**config["model"])
    model.to(device=device)
    model = idist.auto_model(model)

    optimizer = optim.Adam(model.parameters(), **config["optimizer"])
    optimizer = idist.auto_optim(optimizer)
    # optimizer = optim.Adam(
    #     [
    #         {"params": model.audio_enc.parameters(), "name": "audio-enc"},
    #         {"params": model.image_enc.parameters(), "name": "image-enc"},
    #     ],
    #     **config["optimizer"],
    # )

    # Trainer and evaluator
    trainer = create_supervised_trainer(
        model,
        optimizer,
        prepare_batch=UtilsTraining.prepare_batch_fn,
        model_fn=UtilsTraining.model_fn,
        loss_fn=identity_loss,
        device=device,
    )
    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=UtilsTraining.prepare_batch_fn,
        model_fn=UtilsTraining.model_fn,
        device=device,
        metrics=UtilsTraining.get_metrics(device),  # type: ignore
    )
    evaluators = {
        lang: {
            test_name: create_supervised_evaluator(
                model,
                prepare_batch=UtilsPairedTest.prepare_batch_fn,
                model_fn=UtilsPairedTest.model_fn,
                device=device,
                output_transform=UtilsPairedTest.output_transform,
                metrics=UtilsPairedTest.get_metrics(lang, test_name, device),  # type: ignore
            )
            for test_name in TEST_NAMES
        }
        for lang in config["data"]["langs"]
    }

    # Early stopping
    # handler = EarlyStopping(config["patience"], score_func, trainer)
    # evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    metric = RunningAverage(output_transform=lambda x: x)
    metric.attach(trainer, "running-average-loss")

    # Print metrics to the stderr with `add_event_handler` API for training stats
    def print_metrics(engine, tag):
        assert tag == "train"
        print(
            "{:s} ◇ {:s} · {:4d} / {:4d} · loss: {:.3f} · loss avg: {:.3f} ◇ lr: {:f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tag,
                engine.state.epoch,
                engine.state.iteration,
                engine.state.output,
                engine.state.metrics["running-average-loss"],
                optimizer.param_groups[0]["lr"],
            )
        )

    if rank == 0:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=config["log_every_iters"]),
            print_metrics,
            tag="train",
        )

    num_steps_per_epoch = len(dataloader_train)
    warmup_duration = num_steps_per_epoch * config["warmup_epochs"]
    cycle_size = (
        num_steps_per_epoch * (config["max_epochs"] - config["warmup_epochs"]) + 1
    )
    lr_min = 1e-6
    base_scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=config["optimizer"]["lr"],
        end_value=lr_min,
        cycle_size=cycle_size,
    )
    scheduler = create_lr_scheduler_with_warmup(
        base_scheduler,
        warmup_start_value=lr_min,
        warmup_end_value=config["optimizer"]["lr"],
        warmup_duration=warmup_duration,
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    def get_results_per_lang(lang, evaluators):
        return "{} → FF: {:.3f} · NF: {:.3f}".format(
            lang,
            evaluators["familiar-familiar"].state.metrics[
                f"{lang}/accuracy-familiar-familiar"
            ],
            evaluators["novel-familiar"].state.metrics[
                f"{lang}/accuracy-novel-familiar"
            ],
        )

    # Run evaluation at every training epoch end with shortcut `on` decorator
    # API and print metrics to the stderr again with `add_event_handler` API for
    # evaluation stats.
    @trainer.on(Events.STARTED | Events.EPOCH_COMPLETED(every=1))
    def _():
        langs = config["data"]["langs"]
        for lang, dataloaders1 in dataloaders.items():
            for test_name, dataloader in dataloaders1.items():
                evaluators[lang][test_name].run(dataloader)
        evaluator.run(dataloader_valid)
        if rank == 0:
            print(
                "{:s} ◇ {:s} · {:4d} / {:4d} · loss: {:.3f} ◇ {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "eval.",
                    trainer.state.epoch,
                    trainer.state.iteration,
                    evaluator.state.metrics["loss"],
                    " ◇ ".join(
                        get_results_per_lang(lang, evaluators[lang]) for lang in langs
                    ),
                )
            )

    if rank == 0:
        # Create a logger
        log_dir = output_dir / "tb-logs"
        tb_logger = TensorboardLogger(log_dir=log_dir)

        # Attach the logger to the trainer to log training loss at each iteration
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=config["log_every_iters"]),
            tag="train",
            output_transform=lambda loss: {"loss": loss},
        )

        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.COMPLETED,
            tag="valid",
            metric_names=["loss"],
            global_step_transform=global_step_from_engine(trainer),
        )

        for lang, evaluators1 in evaluators.items():
            for test_name, evaluator1 in evaluators1.items():
                tb_logger.attach_output_handler(
                    evaluator1,
                    event_name=Events.COMPLETED,
                    tag="test",
                    metric_names=[f"{lang}/accuracy-{test_name}"],
                    global_step_transform=global_step_from_engine(trainer),
                )

        tb_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_STARTED,
            optimizer=optimizer,
        )

    # Model checkpoint
    def score_func(engine):
        return -engine.state.metrics["loss"]

    model_dir = output_dir / "checkpoints"

    handler = ModelCheckpoint(
        model_dir,
        n_saved=config["n_saved"],
        create_dir=True,
        require_empty=True,
        score_name="neg-loss",
        score_function=score_func,
        save_on_rank=0,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        handler,
        {"model": unwrap_model(model)},
    )

    # Setup is done. Let's run the training.
    trainer.run(
        dataloader_train,
        max_epochs=config["max_epochs"],
        # epoch_length=config["epoch_length"],
    )

    if rank == 0:
        tb_logger.close()


@click.command()
@click.argument("config_name", type=str)
def main(config_name):
    backend = "nccl"
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(train, config_name)


if __name__ == "__main__":
    main()
