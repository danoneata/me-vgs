import sys
from torch import optim
from ignite.handlers import FastaiLRFinder
from ignite.engine import create_supervised_trainer
from mevgs.train import (
    CONFIGS,
    identity_loss,
    model_fn,
    prepare_batch_fn,
    setup_data,
    setup_model,
)

config_name = sys.argv[1]
config = CONFIGS[config_name]

device = config["device"]
model = setup_model(**config["model"])
model.to(device=device)

optimizer = optim.Adam(
    [
        {"params": model.audio_enc.parameters(), "name": "audio-enc"},
        {"params": model.image_enc.parameters(), "name": "image-enc"},
    ],
    **config["optimizer"],
)

trainer = create_supervised_trainer(
    model,
    optimizer,
    prepare_batch=prepare_batch_fn,
    model_fn=model_fn,
    loss_fn=identity_loss,
    device=device,
)

def print_metrics(engine, tag):
    print("{:s} · {:4d} / {:4d} · loss: {:.3f}".format(
            tag,
            engine.state.epoch,
            engine.state.iteration,
            engine.state.output,
        )
    )

trainer.add_event_handler(
    Events.ITERATION_COMPLETED(every=config["log_every_iters"]),
    print_metrics,
    tag="train",
)

dataloader_train, _ = setup_data(**config["data"])