import os
import time
import math
import sys
import json
from matplotlib.pyplot import figure
import glob

import torch
import torch.nn.functional as F
import wandb
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import datetime

import utils
from timm.utils import AverageMeter

# Rich console
CONSOLE = Console(width=1800)


def log_step(
        current_epoch,
        total_epochs,
        current_step,
        total_steps,
        loss,
        times,
        prefix):
    """
    Log metrics to the console after a forward pass

    Args:
        current_epoch: 当前epoch
        total_epochs: 总共epoch
        current_step: 当前steps
        total_steps: 总共steps
        loss: 损失值
        times
        prefix: 打印前缀
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    table.add_column("STEP")
    table.add_column("LOSS")
    for item in times:
        table.add_column(f"{item.upper()} TIME")
    time_values = [f"{t:.2f}" for t in times.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        f"{current_step} / {total_steps}",
        f"{loss:.2f}",
        *tuple(time_values),
    )
    CONSOLE.print(table)


def log_epoch(
        current_epoch,
        total_epochs,
        metrics,
        prefix):
    """
    Log metrics to the console after an epoch

    Args:
        current_epoch: 当前epoch
        total_epochs: 总共的epoch
        metrics: 指标
        prefix: log前缀

    """
    table = Table(show_header=True, header_style="bold", width=200)  # , width=128
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    for k in metrics:
        table.add_column(k.replace(prefix, "").replace("/", "").upper())
    metric_values = [f"{m:.4f}" for m in metrics.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        *tuple(metric_values),
    )
    CONSOLE.print(table)


def train_one_epoch(
        current_epoch,
        total_epochs,
        model,
        optimizer,
        dataloader,
        lr_scheduler=None,
        figures_path=None,
        reduction_method="svd",
        wandb_run=None,
        log_console=True,
        log_by_step=False,
        log_by_epoch=True,
        device="cpu",
):
    """
    Train the given model for one epoch with the given dataloader and optimizer

    Args:
        current_epoch: 当前epoch
        total_epochs: 总epochs
        model: 模型
        optimizer: 模型优化器
        dataloader: 数据加载器
        lr_scheduler: 学习率调整策略
        figures_path: 图片保存路径
        reduction_method: 数据降维方法。默认为svd
        wandb_run: wandb实例
        log_console: 是否打印log
        log_by_step: 单步打印log
        log_by_epoch: 单epoch打印log
        device: 在哪个设备上训练
    """
    # Put the model in training mode
    model.train()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time, epoch_opt_time = 0, 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()

    # 计算剩余运算时长
    num_steps = len(dataloader)
    index = 0
    batch_time = AverageMeter()
    start_time = time.time()

    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time  # 计算一个epoch的运行时间=model_time

        # Get model outputs
        model_time = time.time()  # 模型初始化时间
        embeddings, preds, loss = model(
            spectrograms.to(device), speakers=speakers.to(device)
        )
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss  # 累加每次model的loss
        epoch_data_time += data_time  # 累计epoch时间
        epoch_model_time += model_time  # 累计model时间
        epoch_embeddings += embeddings  # 累计embedding
        epoch_targets += speakers.detach().cpu().tolist()  # 累计speaker_id
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # Perform backpropagation
        opt_time = time.time()  # 反向传播时间+优化时间
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        opt_time = time.time() - opt_time
        epoch_opt_time += opt_time  # 累计反向传播时间+优化时间

        # Log to console
        if log_console and log_by_step:
            now = time.strftime("%m-%d %H:%M:%S")
            times = {"model": model_time, "data": data_time, "opt": opt_time}
            log_step(
                current_epoch, total_epochs, step, len(dataloader), loss, times, now + "train"
            )
        # Empty CUDA cache
        now = time.strftime("%m-%d %H:%M:%S")

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        etas = batch_time.avg * (num_steps - step)
        index += len(speakers)

        times = {"model": f"{model_time:.2f}", "data": f"{data_time:.2f}", "opt": f"{opt_time:.2f}"}
        sys.stderr.write(
            f"{now}: train in epoch: {current_epoch} / {total_epochs}, steps: {step}/{len(dataloader)}, {times}"
            # f"loss: {epoch_loss.detach().cpu().numpy()/step:.5f}, acc: {epoch_preds/index*len(speakers)}"
            f" time left: {datetime.timedelta(seconds=int(etas))}\r")
        sys.stderr.flush()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    sys.stdout.write("\n")

    # Get metrics
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = tools.get_train_val_metrics(
            epoch_targets, epoch_preds, prefix="train"
        )
    metrics["train/total_loss"] = epoch_loss
    metrics["train/avg_loss"] = epoch_loss / len(dataloader)
    metrics["train/total_data_time"] = epoch_data_time
    metrics["train/avg_data_time"] = epoch_data_time / len(dataloader)
    metrics["train/total_model_time"] = epoch_model_time
    metrics["train/avg_model_time"] = epoch_model_time / len(dataloader)
    metrics["train/total_opt_time"] = epoch_opt_time
    metrics["train/avg_opt_time"] = epoch_opt_time / len(dataloader)
    metrics["train/lr"] = (
        lr_scheduler.get_last_lr()[0]
        if lr_scheduler is not None
        else optimizer.param_groups[0]["lr"]
    )

    # Log to console
    if log_console and log_by_epoch:
        log_epoch(current_epoch, total_epochs, metrics, "train")

    # Plot embeddings
    if figures_path is not None:
        figure_path = os.path.join(figures_path, f"epoch_{current_epoch}_train.png")
        tools.visualize_embeddings(
            torch.stack(epoch_embeddings),
            epoch_targets,
            reduction_method=reduction_method,
            show=False,
            save=figure_path,
            only_centroids=False,
        )
        if wandb_run is not None:
            metrics["train/embeddings"] = wandb.Image(figure_path)

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def save_checkpoint(
        epoch, checkpoints_path, model, optimizer, lr_scheduler=None, wandb_run=None
):
    """
    Save the current state of the model, optimizer and learning rate scheduler,
    both locally and on wandb (if available and enabled)

    Args:
        epoch: 当前epoch
        checkpoints_path: 保存路径
        model: 模型
        optimizer: 模型优化器
        lr_scheduler: 学习率调整策略
        wandb_run: wandb实例
    """
    # Create state dictionary
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else dict()
        ),
        "epoch": epoch,
    }

    # Save state dictionary
    checkpoint_file = os.path.join(checkpoints_path, f"epoch_{epoch}.pth")
    torch.save(state_dict, checkpoint_file)
    print(f"{checkpoint_file} was saved!")
    if wandb_run is not None:
        pass
        # TODO Linux应该能正常save，windows如何save？
        # wandb_run.save(checkpoint_file)


def load_last_checkpoint(model, optimizer, lr_scheduler, checkpoint_root_path):
    # epoch, checkpoints_path, model, optimizer, lr_scheduler = None, wandb_run = None
    # 检查现有模型并加载最后一个模型
    # 参考自：https: // zhuanlan.zhihu.com / p / 82038049
    model_files = glob.glob('%s/epoch_*.pth' % checkpoint_root_path)
    model_files.sort(key=os.path.getmtime)
    if len(model_files) >= 1:
        print("load from previous model %s!" % model_files[-1])
        next_epoch = int(os.path.splitext(os.path.basename(model_files[-1]))[0][6:]) + 1
    else:
        raise Exception("Can not find initial model")
    loaded_state = torch.load(model_files[-1])

    model.load_state_dict(loaded_state['model'])
    optimizer.load_state_dict(loaded_state['optimizer'])
    lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
    epoch = loaded_state['epoch']
    # loss = loaded_state['loss']

    return model, optimizer, lr_scheduler, epoch


def load_single_checkpoint(model, checkpoint_path):
    """
    test用的加载函数
    """
    loaded_state = torch.load(checkpoint_path)
    model.load_state_dict(loaded_state['model'])
    # optimizer.load_state_dict(loaded_state['optimizer'])
    # lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
    epoch = loaded_state['epoch']
    return model, epoch


def training_loop(
        run_name,
        last_epoch,
        epochs,
        model,
        optimizer,
        train_dataloader,
        checkpoints_path,
        test_dataset=None,
        val_dataloader=None,
        val_every=None,
        figures_path=None,
        reduction_method="svd",
        lr_scheduler=None,
        checkpoints_frequency=None,
        wandb_run=None,
        log_console=True,
        mindcf_p_target=0.01,
        mindcf_c_fa=1,
        mindcf_c_miss=1,
        device="cpu",
):
    """
    Standard training loop function: train and evaluate
    after each training epoch
    """
    # Create checkpoints directory
    checkpoints_path = os.path.join(checkpoints_path, run_name)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Create figures directory
    if figures_path is not None:
        figures_path = os.path.join(figures_path, run_name)
        os.makedirs(figures_path, exist_ok=True)

    # For each epoch
    start_epoch = last_epoch + 1
    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        # Train for one epoch
        train_one_epoch(
            epoch,
            epochs,
            model,
            optimizer,
            train_dataloader,
            lr_scheduler=lr_scheduler,
            figures_path=figures_path,
            reduction_method=reduction_method,
            wandb_run=wandb_run,
            log_console=log_console,
            device=device,
        )
        end = time.time()
        now = time.strftime("%m-%d %H:%M:%S")
        print(f"epoch {epoch} total training time is "
              f"{int(end - start) // 3600}"
              f":{int(end - start) % 3600 // 60}"
              f":{int(end - start) % 60}  at {now}\r")
        # Decay the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoints once in a while
        if checkpoints_frequency is not None and epoch % checkpoints_frequency == 0:
            save_checkpoint(
                epoch,
                checkpoints_path,
                model,
                optimizer,
                lr_scheduler=lr_scheduler,
                wandb_run=wandb_run,
            )

        # Evaluate once in a while (always evaluate at the first and last epochs)
        if (
                val_dataloader is not None
                and val_every is not None
                and (epoch % val_every == 0 or epoch == 1 or epoch == epochs)
        ):
            evaluate(
                model,
                val_dataloader,
                current_epoch=epoch,
                total_epochs=epochs,
                figures_path=figures_path,
                reduction_method=reduction_method,
                wandb_run=wandb_run,
                log_console=log_console,
                device=device,
            )

    # Always save the last checkpoint
    save_checkpoint(
        epochs,
        checkpoints_path,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        wandb_run=wandb_run,
    )

    # Final test
    if test_dataset is not None:
        test(
            model,
            test_dataset,
            wandb_run=wandb_run,
            log_console=log_console,
            mindcf_p_target=mindcf_p_target,
            mindcf_c_fa=mindcf_c_fa,
            mindcf_c_miss=mindcf_c_miss,
            device=device,
        )


@torch.no_grad()
def evaluate(
        model,
        dataloader,
        current_epoch=None,
        total_epochs=None,
        figures_path=None,
        figure_name=None,
        reduction_method="svd",
        wandb_run=None,
        log_console=True,
        log_by_step=False,
        log_by_epoch=True,
        device="cpu",
):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time = 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()
    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(
            spectrograms.to(device), speakers=speakers.to(device)
        )
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_data_time += data_time
        epoch_model_time += model_time
        epoch_embeddings += embeddings
        epoch_targets += speakers.detach().cpu().tolist()
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log to console
        if log_console and log_by_step:
            times = {"model": model_time, "data": data_time}
            log_step(
                current_epoch, total_epochs, step, len(dataloader), loss, times, "val"
            )

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    # Get metrics and return them
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = tools.get_train_val_metrics(epoch_targets, epoch_preds, prefix="val")
    metrics[f"val/total_loss"] = epoch_loss
    metrics[f"val/avg_loss"] = epoch_loss / len(dataloader)
    metrics[f"val/total_data_time"] = epoch_data_time
    metrics[f"val/avg_data_time"] = epoch_data_time / len(dataloader)
    metrics[f"val/total_model_time"] = epoch_model_time
    metrics[f"val/avg_model_time"] = epoch_model_time / len(dataloader)

    # Log to console
    if log_console and log_by_epoch:
        log_epoch(current_epoch, total_epochs, metrics, "val")

    # Plot embeddings
    if figures_path is not None:
        if figure_name is None:
            figure_name = f"epoch_{current_epoch}_val.png"
        figure_path = os.path.join(figures_path, figure_name)
        tools.visualize_embeddings(
            torch.stack(epoch_embeddings),
            epoch_targets,
            reduction_method=reduction_method,
            show=False,
            save=figure_path,
            only_centroids=False,
        )
        if wandb_run is not None:
            metrics[f"val/embeddings"] = wandb.Image(figure_path)

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


@torch.no_grad()
def test(
        model,
        test_dataset,
        indices=None,
        wandb_run=None,
        log_console=True,
        log_by_step=False,
        log_by_epoch=True,
        mindcf_p_target=0.01,
        mindcf_c_fa=1,
        mindcf_c_miss=1,
        device="cpu",
):
    """
    Test the given model and store EER and minDCF metrics
    """
    # Put the model in evaluation mode
    model.eval()

    # Get cosine similarity scores and labels
    samples = (
        test_dataset.get_sample_pairs(indices=indices, device=device)
        if not isinstance(test_dataset, torch.utils.data.Subset)
        else test_dataset.dataset.get_sample_pairs(
            indices=test_dataset.indices, device=device
        )
    )
    scores, labels = [], []
    for s1, s2, label in tqdm(samples, desc="Building scores and labels"):
        e1, e2 = model(s1), model(s2)
        scores += [F.cosine_similarity(e1, e2).item()]
        labels += [int(label)]

    # Get test metrics (EER and minDCF)
    metrics = tools.get_test_metrics(
        scores,
        labels,
        mindcf_p_target=mindcf_p_target,
        mindcf_c_fa=mindcf_c_fa,
        mindcf_c_miss=mindcf_c_miss,
        prefix="test",
    )

    # Log to console
    if log_console and log_by_epoch:
        log_epoch(None, None, metrics, "test")

    # Log to wandb
    if wandb_run is not None:
        wandb_run.notes = json.dumps(metrics, indent=2).encode("utf-8")

    return metrics


def test_loop(
        model,
        test_dataset,
        checkpoint_root_path,
        log_name="score.txt",
        wandb_run=None,
        log_console=True,
        log_by_step=False,
        log_by_epoch=False,
        mindcf_p_target=0.01,
        mindcf_c_fa=1,
        mindcf_c_miss=1,
        device="cpu",
):
    model_files = glob.glob('%s/epoch_*.pth' % checkpoint_root_path)
    model_files.sort(key=os.path.getmtime)
    if len(model_files) == 0:
        raise Exception("Can not find initial model")
    loaded_state = torch.load(model_files[-1])
    log_path = os.path.join(checkpoint_root_path, log_name)
    score_file = open(log_path, "a+")
    for model_file in model_files:
        model, epoch = load_single_checkpoint(model, model_file)
        metrics = test(model,
                       test_dataset,
                       wandb_run=wandb_run,
                       log_console=log_console,
                       log_by_step=log_by_step,
                       log_by_epoch=log_by_epoch,
                       mindcf_p_target=mindcf_p_target,
                       mindcf_c_fa=mindcf_c_fa,
                       mindcf_c_miss=mindcf_c_miss,
                       device=device,
                       )
        print(f"epoch {epoch}, "+metrics)
        score_file.write(f"epoch {epoch}, eer={metrics['test/eer']:.4f}, min_dcf={metrics['test/mindcf']:.4f}\n")
        score_file.flush()
    score_file.close()


def transform_model(model, checkpoint_dir_path, device="cpu"):
    loaded_state = torch.load(checkpoint_dir_path)
    model.load_state_dict(loaded_state['model'])
    example = torch.randn(1, 80, 301)
    example = example.to(device)
    model = model.to(device)
    model.eval()
    traced_script_model = torch.jit.trace(model, example)
    traced_script_model.save("baseline_epoch_78_gpu.pt")


def infer(
        model,
        utterances,
        speakers,
        dataset,
        reduction_method="svd",
        figure_path=None,
        device="cpu",
):
    """
    Compute embeddings for the given utterances and plot them
    """
    # Put the model in evaluation mode
    model.eval()

    # Compute embeddings
    all_embeddings = []
    for utterance in tqdm(utterances):
        data = dataset[utterance]
        embeddings = model(data["spectrogram"].to(device))
        all_embeddings += embeddings

    # Show embeddings
    tools.visualize_embeddings(
        torch.stack(all_embeddings),
        speakers,
        reduction_method=reduction_method,
        show=True,
        save=figure_path,
        only_centroids=False,
    )
