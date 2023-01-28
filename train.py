import os
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import torch
import torch.optim as optim
import yaml

import models, losses, transforms, utils, datasets, learn


def train(params, only_test=False, transform=False):
    """
    Training loop entry-point
    """
    tools.set_seed(params.generic.seed)
    device = tools.get_device()
    # Set number of threads
    if params.generic.workers > 0:
        torch.set_num_threads(params.generic.workers)

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

    train_transformations = partial_get_transforms(training=True)
    non_train_transformations = partial_get_transforms(training=False)

    # Get datasets and dataloaders
    train_dataset, val_dataset, test_dataset, n_speakers = datasets.get_datasets(
        params.dataset.root,
        train_transformations=train_transformations,
        non_train_transformations=non_train_transformations,
        val=params.validation.enabled,
        val_utterances_per_speaker=params.validation.num_utterances_per_speaker,
        test=params.test.enabled,
        test_speakers=params.test.num_speakers,
        test_utterances_per_speaker=params.test.num_utterances_per_speaker,
    )
    if params.dumb.enabled:
        train_dataset = test_dataset
    train_dataloader = datasets.get_dataloader(
        train_dataset,
        batch_size=params.training.batch_size,
        shuffle=True,
        num_workers=params.generic.workers,
        n_mels=params.audio.spectrogram.n_mels,
        seed=params.generic.seed,
    )
    val_dataloader = datasets.get_dataloader(
        val_dataset,
        batch_size=params.validation.batch_size,
        shuffle=False,
        num_workers=params.generic.workers,
        n_mels=params.audio.spectrogram.n_mels,
        seed=params.generic.seed,
    )

    run_name = ""
    last_epoch = 0

    # Get loss function
    loss_params = dict()
    if params.training.loss in params.loss.__dict__:
        loss_params = params.loss.__dict__[params.training.loss].__dict__["entries"]
    loss_function = losses.LOSSES[params.training.loss](
        params.generic.embedding_size, n_speakers, device=device, **loss_params
    )

    # Get model
    if params.dumb.enabled:
        model = titanetmodel.DumbConvNet(
            params.audio.spectrogram.n_mels,
            loss_function=loss_function,
            hidden_size=params.dumb.hidden_size,
            embedding_size=params.generic.embedding_size,
            n_layers=params.dumb.n_layers,
            device=device,
        )
    elif params.baseline.enabled:
        model = titanetmodel.DVectorBaseline(
            params.audio.spectrogram.n_mels,
            loss_function=loss_function,
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

        model = titanetmodel.TitaNet.get_titanet(
            embedding_size=params.generic.embedding_size,
            n_mels=params.audio.spectrogram.n_mels,
            n_mega_blocks=n_mega_blocks,
            model_size=params.titanet.model_size,
            attention_hidden_size=params.titanet.attention_hidden_size,
            simple_pool=params.titanet.simple_pool,
            loss_function=loss_function,
            dropout=params.titanet.dropout,
            device=device,
        )

    # Get optimizer and scheduler
    optimizer_type = optim.SGD if params.training.optimizer == "sgd" else optim.Adam
    optimizer = optimizer_type(
        model.parameters(),
        lr=params.training.optimizer.start_lr,
        weight_decay=params.training.optimizer.weight_decay,
    )

    optimizer = tools.optimizer_to(optimizer, device=device)
    lr_scheduler = None
    if params.training.optimizer.scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.training.epochs,
            eta_min=params.training.optimizer.end_lr,
        )
        lr_scheduler = tools.scheduler_to(lr_scheduler, device=device)

    # 是否记载之前的模型
    if params.titanet.load_checkpoint:
        run_name = params.titanet.checkpoint_path
        checkpoint_path = os.path.join(params.training.checkpoints_path, run_name)
        model, optimizer, lr_scheduler, last_epoch = \
            learn.load_last_checkpoint(model, optimizer, lr_scheduler, checkpoint_path + "/")

    # Use backprop to chart dependencies
    if params.generic.chart_dependencies:
        tools.chart_dependencies(
            model, n_mels=params.audio.spectrogram.n_mels, device=device
        )

    # Start wandb logging
    wandb_run = None
    run_name = run_name if run_name != "" else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Initialize scheduler at {run_name}")
    if params.wandb.enabled:
        wandb_run = tools.init_wandb(
            params.wandb.api_key_file,
            params.wandb.project,
            params.wandb.entity,
            name=run_name,
            config=params.entries,
        )
        print(f"Initialize wan_db successful")

    if not only_test and not transform:
        # Perform training loop
        learn.training_loop(
            run_name,
            last_epoch,
            params.training.epochs,
            model,
            optimizer,
            train_dataloader,
            params.training.checkpoints_path,
            test_dataset=test_dataset if params.test.enabled else None,
            val_dataloader=val_dataloader,
            val_every=params.validation.every if params.validation.enabled else None,
            figures_path=params.figures.path+run_name if params.figures.enabled else None,  # figure_path+run_name
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
    if only_test and not transform:
        learn.test_loop(
            model,
            test_dataset,
            params.training.checkpoints_path+params.test.direct_path,
            log_name=params.test.log_name,
            wandb_run=wandb_run,
            log_console=params.generic.log_console,
            mindcf_p_target=params.test.mindcf_p_target,
            mindcf_c_fa=params.test.mindcf_c_fa,
            mindcf_c_miss=params.test.mindcf_c_miss,
            device=device,
        )

    if transform:
        learn.transform_model(model, "../exps/tita_exp1/2023-01-03_17-04-39/epoch_78.pth", device=device)

    # Stop wandb logging
    if params.wandb.enabled:
        wandb_run.finish()
    print(f"The training is end!")


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
    print(args.params)
    with open(args.params, "r", encoding='UTF-8') as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    params = tools.Struct(**args)

    # Call the training function
    train(params)
    # train(params, only_test=True)
    # train(params, transform=True)
