CONFIGS = {
    "00": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 50,
        # "epoch_length": 500,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 32,
            "num_neg": 256,
            "num_workers": 32,
            "num_word_repeats": 64,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "01": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 50,
        # "epoch_length": 500,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 32,
            "num_neg": 256,
            "num_workers": 32,
            "num_word_repeats": 64,
            "to_shuffle": False,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "02": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 5,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 8,
            "num_neg": 32,
            "num_workers": 32,
            "to_shuffle": False,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "03": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 5,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 9,
            "num_workers": 32,
            "batch_size": 12,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "04": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "05": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "06": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "07": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "08": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "09": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "10": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 32,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 48,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "11": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 32,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 120,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "12": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 36,
        "warmup_epochs": 6,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 120,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "13": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 36,
        "warmup_epochs": 6,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 120,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "14": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "15": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "16": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "num_channels": 512,
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
                "use_pretrained_cpc": True,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "17": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "mattnet",
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "output_dim": 2048,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "18": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 8e-5,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "mattnet",
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "output_dim": 2048,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "19": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 8e-5,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "mattnet",
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "output_dim": 2048,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "20": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 8e-5,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "mattnet",
            "embed_dim": 2048,
            "pooling": "features-avg",
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
            },
            "image_encoder_kwargs": {
                "use_pretrained_alexnet": True,
                "to_freeze_backbone": True,
            },
        },
    },
    "21": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 8e-5,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 2048,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
            },
            "image_encoder_kwargs": {
                "use_pretrained_alexnet": True,
                "to_freeze_backbone": True,
            },
        },
    },
    "22": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 2048,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
            },
            "image_encoder_kwargs": {
                "backbone_type": "dino-resnet50",
                "to_freeze_backbone": True,
                "use_pretrained_backbone": True,
            },
        },
    },
    "23": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 2048,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "backbone_type": "alexnet",
                "to_freeze_backbone": True,
                "use_pretrained_backbone": True,
                "width": 256,
            },
        },
    },
    "24": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 34,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 4e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 16,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "backbone_type": "dino-resnet50",
                "to_freeze_backbone": True,
                "use_pretrained_backbone": True,
                "width": 256,
            },
        },
    },
    "test": {
        "seed": 1337,
        "device": "cuda",
        "max_epochs": 2,
        "warmup_epochs": 1,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "backbone_type": "alexnet",
                "to_freeze_backbone": True,
                "use_pretrained_backbone": True,
                "width": 256,
            },
        },
    },
}

# config with multiple seeds
for seed, v in enumerate("abcde"):
    CONFIGS[f"25{v}"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type": "wavlm-base-plus",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "backbone_type": "dino-resnet50",
                "to_freeze_backbone": True,
                "use_pretrained_backbone": True,
                "width": 256,
            },
        },
    }


CONFIGS[f"25a-redo"] = {
    "seed": 0,
    "device": "cuda",
    "max_epochs": 24,
    "warmup_epochs": 4,
    "n_saved": 5,
    "log_every_iters": 5,
    "optimizer": {
        "lr": 2e-4,
        "weight_decay": 5e-7,
    },
    "data": {
        "feature_type_audio": "wavlm-base-plus",
        "feature_type_image": "raw",
        "langs": ("english",),
        "num_pos": 1,
        "num_neg": 11,
        "num_workers": 12,
        "batch_size": 60,
    },
    "model": {
        "model_name": "clip",
        "embed_dim": 256,
        "audio_encoder_kwargs": {
            "type": "transformer",
            "input_dim": 768,
            "width": 256,
        },
        "image_encoder_kwargs": {
            "backbone_type": "dino-resnet50",
            "to_freeze_backbone": True,
            "use_pretrained_backbone": True,
            "width": 256,
        },
    },
}


for seed, v in enumerate("abcde"):
    CONFIGS[f"26{v}"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type_audio": "wavlm-base-plus",
            "feature_type_image": "dino-resnet50",
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "input_dim": 2048,
                "width": 256,
                "backbone_type": "identity",
                "to_freeze_backbone": None,
                "use_pretrained_backbone": None,
            },
        },
    }


LANG_SHORT = {
    "english": "en",
    "dutch": "nl",
    "french": "fr",
}


for lang in ("dutch", "french"):
    for seed, v in enumerate("abcde"):
        lang_short = LANG_SHORT[lang]
        CONFIGS[f"26{v}-{lang_short}"] = {
            "seed": seed,
            "device": "cuda",
            "max_epochs": 24,
            "warmup_epochs": 4,
            "n_saved": 5,
            "log_every_iters": 5,
            "optimizer": {
                "lr": 2e-4,
                "weight_decay": 5e-7,
            },
            "data": {
                "feature_type_audio": "wavlm-base-plus",
                "feature_type_image": "dino-resnet50",
                "langs": (lang,),
                "num_pos": 1,
                "num_neg": 11,
                "num_workers": 12,
                "batch_size": 60,
            },
            "model": {
                "model_name": "clip",
                "embed_dim": 256,
                "audio_encoder_kwargs": {
                    "type": "transformer",
                    "input_dim": 768,
                    "width": 256,
                },
                "image_encoder_kwargs": {
                    "input_dim": 2048,
                    "width": 256,
                    "backbone_type": "identity",
                    "to_freeze_backbone": None,
                    "use_pretrained_backbone": None,
                },
            },
        }


for seed, v in enumerate("abcde"):
    CONFIGS[f"26{v}-en-fr"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type_audio": "wavlm-base-plus",
            "feature_type_image": "dino-resnet50",
            "langs": ("english", "french"),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "input_dim": 2048,
                "width": 256,
                "backbone_type": "identity",
                "to_freeze_backbone": None,
                "use_pretrained_backbone": None,
            },
        },
    }

for seed, v in enumerate("abcde"):
    CONFIGS[f"27{v}-en-fr"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type_audio": "wavlm-base-plus",
            "feature_type_image": "dino-resnet50",
            "langs": ("english", "french"),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 512,
            },
            "image_encoder_kwargs": {
                "input_dim": 2048,
                "width": 256,
                "backbone_type": "identity",
                "to_freeze_backbone": None,
                "use_pretrained_backbone": None,
            },
        },
    }

for seed, v in enumerate("abcde"):
    CONFIGS[f"28{v}-en-fr"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type_audio": "wavlm-base-plus",
            "feature_type_image": "dino-resnet50",
            "langs": ("english", "french"),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 256,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 256,
            },
            "image_encoder_kwargs": {
                "input_dim": 2048,
                "width": 512,
                "backbone_type": "identity",
                "to_freeze_backbone": None,
                "use_pretrained_backbone": None,
            },
        },
    }


for seed, v in enumerate("abcde"):
    CONFIGS[f"29{v}-en-fr"] = {
        "seed": seed,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 4,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "feature_type_audio": "wavlm-base-plus",
            "feature_type_image": "dino-resnet50",
            "langs": ("english", "french"),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 12,
            "batch_size": 60,
        },
        "model": {
            "model_name": "clip",
            "embed_dim": 512,
            "audio_encoder_kwargs": {
                "type": "transformer",
                "input_dim": 768,
                "width": 512,
            },
            "image_encoder_kwargs": {
                "input_dim": 2048,
                "width": 512,
                "backbone_type": "identity",
                "to_freeze_backbone": None,
                "use_pretrained_backbone": None,
            },
        },
    }


SIZES = {
    "lg": 512,
    "sm": 128,
    "xsm": 64,
}


for s in ["xsm", "sm", "lg"]:
    for langs in [("english", ), ("french", ), ("english", "french")]:
        for seed, v in enumerate("abcde"):
            e = SIZES[s]
            suffix = "-".join(LANG_SHORT[l] for l in langs)
            CONFIGS[f"26{v}-{s}-{suffix}"] = {
                "seed": seed,
                "device": "cuda",
                "max_epochs": 24,
                "warmup_epochs": 4,
                "n_saved": 5,
                "log_every_iters": 5,
                "optimizer": {
                    "lr": 2e-4,
                    "weight_decay": 5e-7,
                },
                "data": {
                    "feature_type_audio": "wavlm-base-plus",
                    "feature_type_image": "dino-resnet50",
                    "langs": langs,
                    "num_pos": 1,
                    "num_neg": 11,
                    "num_workers": 12,
                    "batch_size": 60,
                },
                "model": {
                    "model_name": "clip",
                    "embed_dim": e,
                    "audio_encoder_kwargs": {
                        "type": "transformer",
                        "input_dim": 768,
                        "width": e,
                    },
                    "image_encoder_kwargs": {
                        "input_dim": 2048,
                        "width": e,
                        "backbone_type": "identity",
                        "to_freeze_backbone": None,
                        "use_pretrained_backbone": None,
                    },
                },
            }


for langs in [("english", ), ("french", ), ("english", "french")]:
    for seed, v in enumerate("abcde"):
        suffix = "-".join(LANG_SHORT[l] for l in langs)
        CONFIGS[f"barlip-00{v}-{suffix}"] = {
            "seed": seed,
            "device": "cuda",
            "max_epochs": 24,
            "warmup_epochs": 4,
            "n_saved": 5,
            "log_every_iters": 5,
            "optimizer": {
                "lr": 4e-5,
                "weight_decay": 5e-7,
            },
            "data": {
                "feature_type_audio": "wavlm-base-plus",
                "feature_type_image": "dino-resnet50",
                "langs": langs,
                "num_workers": 12,
                "batch_size": 768,
            },
            "model": {
                "model_name": "barlip",
                "embed_dim": 256,
                "λ": 0.005,
                "audio_encoder_kwargs": {
                    "type": "transformer",
                    "input_dim": 768,
                    "width": 256,
                },
                "image_encoder_kwargs": {
                    "input_dim": 2048,
                    "width": 256,
                    "backbone_type": "identity",
                    "to_freeze_backbone": None,
                    "use_pretrained_backbone": None,
                },
            },
        }

for langs in [("english", ), ("french", ), ("english", "french")]:
    for size in ["sm", "lg"]:
        for seed, v in enumerate("abcde"):
            e = SIZES[size]
            # b = 768 if size == "sm" else 704
            b = 768
            suffix = "-".join(LANG_SHORT[l] for l in langs)
            CONFIGS[f"barlip-00{v}-{size}-{suffix}"] = {
                "seed": seed,
                "device": "cuda",
                "max_epochs": 24,
                "warmup_epochs": 4,
                "n_saved": 5,
                "log_every_iters": 5,
                "optimizer": {
                    "lr": 4e-5,
                    "weight_decay": 5e-7,
                },
                "data": {
                    "feature_type_audio": "wavlm-base-plus",
                    "feature_type_image": "dino-resnet50",
                    "langs": langs,
                    "num_workers": 12,
                    "batch_size": b,
                },
                "model": {
                    "model_name": "barlip",
                    "embed_dim": e,
                    "λ": 0.005,
                    "audio_encoder_kwargs": {
                        "type": "transformer",
                        "input_dim": 768,
                        "width": e,
                    },
                    "image_encoder_kwargs": {
                        "input_dim": 2048,
                        "width": e,
                        "backbone_type": "identity",
                        "to_freeze_backbone": None,
                        "use_pretrained_backbone": None,
                    },
                },
            }
