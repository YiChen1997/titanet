import sys
from argparse import ArgumentParser
from functools import partial

import torch
import torch.optim as optim
import numpy as np
import yaml

import datasets, transforms, models, learn, losses, utils


def get_datasets(
    dataset_root,
    train_transformations=None,
    non_train_transformations=None,
    val_enabled=True,
    val_utterances_per_speaker=10,
    test_enabled=True,
    test_speakers=10,
    test_utterances_per_speaker=10,
):
    """
    Return an instance of the dataset specified in the given
    parameters, splitted into training, validation and test sets
    """
    # Get the dataset
    dataset = datasets.LibriSpeechDataset(
        dataset_root,
        train_transforms=train_transformations,
        non_train_transforms=non_train_transformations,
    )

    # Get speakers
    utterances = dataset.speakers_utterances
    speakers = dataset.speakers

    # Compute train, validation and test utterances
    train_utterances, val_utterances, test_utterances = [], [], []
    for i, s in enumerate(speakers):
        train_start_utterance = 0
        if val_enabled:
            val_utterances += utterances[s][:val_utterances_per_speaker]
            train_start_utterance = val_utterances_per_speaker
        if test_enabled and i < test_speakers:
            test_utterances += utterances[s][
                val_utterances_per_speaker:test_utterances_per_speaker
            ]
            train_start_utterance = test_utterances_per_speaker
        train_utterances += utterances[s][train_start_utterance:]

    # Split dataset
    train_dataset = torch.utils.data.Subset(dataset, train_utterances)
    val_dataset = torch.utils.data.Subset(dataset, val_utterances)
    val_dataset.training = False
    test_dataset = torch.utils.data.Subset(dataset, test_utterances)
    test_dataset.training = False

    return train_dataset, val_dataset, test_dataset, len(speakers)


def get_random_dataloader(dataset, batch_size, num_workers=4, n_mels=80, seed=42):
    """
    Return a dataloader that randomly samples a batch of data
    from the given dataset, to use in the training procedure
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(datasets.collate_fn, n_mels=n_mels),
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=True,
    )


def get_sequential_dataloader(dataset, num_workers=4, n_mels=80, seed=42):
    """
    Return a dataloader that sequentially samples one observation
    at a time from the given dataset, to use in the validation phase
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(datasets.collate_fn, n_mels=n_mels),
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=True,
    )


def get_dataloaders(
    train_dataset, val_dataset, batch_size, num_workers=4, n_mels=80, seed=42
):
    """
    Return the appropriate dataloader for each dataset split
    """
    return (
        get_random_dataloader(
            train_dataset, batch_size, num_workers=num_workers, n_mels=n_mels, seed=seed
        ),
        get_sequential_dataloader(
            val_dataset, num_workers=num_workers, n_mels=n_mels, seed=seed
        ),
    )


def train(params):
    """
    Training loop entry-point
    """
    # Set random seed for reproducibility
    # and get GPU if present
    utils.set_seed(params.generic.seed)
    device = utils.get_device()

    # Set number of threads
    if params.generic.workers > 0:
        torch.set_num_threads(params.generic.workers)

    # Get data transformations
    partial_get_transforms = partial(
        transforms.get_transforms,
        params.augmentation.enable,
        params.augmentation.rir.corpora_path,
        max_length=params.augmentation.chunk.max_length,
        chunk_lengths=params.augmentation.chunk.lengths,
        min_speed=params.augmentation.speed.min,
        max_speed=params.augmentation.speed.max,
        sample_rate=params.audio.sample_rate,
        n_fft=params.audio.spectrogram.n_fft,
        win_length=params.audio.spectrogram.win_length,
        hop_length=params.audio.spectrogram.hop_length,
        n_mels=params.audio.spectrogram.n_mels,
        freq_mask_ratio=params.augmentation.specaugment.freq_mask_ratio,
        freq_mask_num=params.augmentation.specaugment.freq_mask_num,
        time_mask_ratio=params.augmentation.specaugment.time_mask_ratio,
        time_mask_num=params.augmentation.specaugment.time_mask_num,
        probability=params.augmentation.probability,
        device=device,
    )
    train_transformations = partial_get_transforms(inference=False)
    non_train_transformations = partial_get_transforms(inference=True)

    # Get datasets and dataloaders
    train_dataset, val_dataset, test_dataset, n_speakers = get_datasets(
        params.dataset.root,
        train_transformations=train_transformations,
        non_train_transformations=non_train_transformations,
        val_enabled=params.validation.enabled,
        val_utterances_per_speaker=params.validation.num_utterances_per_speaker,
        test_enabled=params.test.enabled,
        test_speakers=params.test.num_speakers,
        test_utterances_per_speaker=params.test.num_utterances_per_speaker,
    )
    if params.dumb.enabled:
        train_dataset = test_dataset
    train_dataloader, val_dataloader = get_dataloaders(
        train_dataset,
        val_dataset,
        params.training.batch_size,
        num_workers=params.generic.workers,
        n_mels=params.audio.spectrogram.n_mels,
        seed=params.generic.seed,
    )

    # Get loss function
    loss_params = dict()
    if params.training.loss in params.loss.__dict__:
        loss_params = params.loss.__dict__[params.training.loss].__dict__["entries"]
    loss_function = losses.LOSSES[params.training.loss](
        params.generic.embedding_size, n_speakers, device=device, **loss_params
    )

    # Get model
    if params.dumb.enabled:
        model = models.DumbConvNet(
            params.audio.spectrogram.n_mels,
            loss_function,
            hidden_size=params.dumb.hidden_size,
            embedding_size=params.generic.embedding_size,
            n_layers=params.dumb.n_layers,
            device=device,
        )
    elif params.baseline.enabled:
        model = models.DVectorBaseline(
            params.audio.spectrogram.n_mels,
            loss_function,
            n_lstm_layers=params.baseline.n_layers,
            hidden_size=params.baseline.hidden_size,
            lstm_average=params.baseline.average,
            embedding_size=params.generic.embedding_size,
            segment_length=params.baseline.segment_length,
            device=device,
        )
    else:
        n_mega_blocks = None
        if params.titanet.n_mega_blocks:
            n_mega_blocks = params.titanet.n_mega_blocks
        model = models.TitaNet.get_titanet(
            loss_function,
            embedding_size=params.generic.embedding_size,
            n_mels=params.audio.spectrogram.n_mels,
            n_mega_blocks=n_mega_blocks,
            model_size=params.titanet.model_size,
            attention_hidden_size=params.titanet.attention_hidden_size,
            simple_pool=params.titanet.simple_pool,
            dropout=params.titanet.dropout,
            device=device,
        )

    # Use backprop to chart dependencies
    if params.generic.chart_dependencies:
        utils.chart_dependencies(
            model, n_mels=params.audio.spectrogram.n_mels, device=device
        )

    # Get optimizer and scheduler
    optimizer_type = optim.SGD if params.training.optimizer == "sgd" else optim.Adam
    optimizer = optimizer_type(
        model.parameters(),
        lr=params.training.optimizer.start_lr,
        weight_decay=params.training.optimizer.weight_decay,
    )
    optimizer = utils.optimizer_to(optimizer, device=device)
    lr_scheduler = None
    if params.training.optimizer.scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.training.epochs,
            eta_min=params.training.optimizer.end_lr,
        )
        lr_scheduler = utils.scheduler_to(lr_scheduler, device=device)

    # Start wandb logging
    wandb_run = None
    run_name = utils.now()
    if params.wandb.enabled:
        wandb_run = utils.init_wandb(
            params.wandb.api_key_file,
            params.wandb.project,
            params.wandb.entity,
            name=run_name,
            config=params.entries,
        )

    # Perform training loop
    learn.training_loop(
        run_name,
        params.training.epochs,
        model,
        optimizer,
        train_dataloader,
        params.training.checkpoints_path,
        test_dataset=test_dataset if params.test.enabled else None,
        val_dataloader=val_dataloader,
        val_every=params.validation.every if params.validation.enabled else None,
        figures_path=params.figures.path if params.figures.enabled else None,
        reduction_method=params.figures.reduction_method,
        lr_scheduler=lr_scheduler,
        checkpoints_frequency=params.training.checkpoints_frequency,
        wandb_run=wandb_run,
        log_console=params.generic.log_console,
        mindcf_p_target=params.test.mindcf_p_target,
        mindcf_c_fa=params.test.mindcf_c_fa,
        mindcf_c_miss=params.test.mindcf_c_miss,
        device=device,
    )

    # Stop wandb logging
    if params.wandb.enabled:
        wandb_run.finish()


if __name__ == "__main__":
    # Parse parameters
    parser = ArgumentParser(description="train the TitaNet model")
    parser.add_argument(
        "-p",
        "--params",
        help="path for the parameters .yml file",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    with open(args.params, "r") as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    params = utils.Struct(**args)

    # Call the training function
    train(params)
