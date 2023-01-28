"""
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
"""
import hashlib
import numpy
import os
import random
import torch
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import umap
import wandb
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from tqdm import tqdm


def init_args(args, score_name='score.txt', model_name='model'):
    args.score_save_path = os.path.join(args.save_path, score_name)
    args.model_save_path = os.path.join(args.save_path, model_name)
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tune_threshold_from_score(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest,
    # and also get the corresponding indexes of the sorted scores.
    # We will treat the sorted scores as the thresholds at which the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    false_negative_rates = []
    false_positive_rates = []

    # At the end of this loop, false_negative_rates[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, false_positive_rates[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            false_negative_rates.append(labels[i])
            false_positive_rates.append(1 - labels[i])
        else:
            false_negative_rates.append(false_negative_rates[i - 1] + labels[i])
            false_positive_rates.append(false_positive_rates[i - 1] + 1 - labels[i])
    false_negative_rates_norm = sum(labels)
    false_positive_rates_norm = len(labels) - false_negative_rates_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    false_negative_rates = [x / float(false_negative_rates_norm) for x in false_negative_rates]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    false_positive_rates = [1 - x / float(false_positive_rates_norm) for x in false_positive_rates]
    return false_negative_rates, false_positive_rates, thresholds


# Computes the minimum of the detection cost function.
# The comments refer to equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(false_negative_rates, false_positive_rates, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(false_negative_rates)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * false_negative_rates[i] * p_target + c_fa * false_positive_rates[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def get_device():
    """
    Return a CUDA device, if available, or a standard CPU device otherwise
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"seed set as {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def overlap(a1, a2):
    """
    Check if the given arrays have common elements
    """
    return len(set(a1).intersection(set(a2))) > 0


def download_auth_url_to_file(
        url, file_path, username, password, hash_prefix=None, progress=True
):
    """
    Download the file at the given URL using the given credentials,
    and finally double-check the checksum of the downloaded file
    """
    if hash_prefix is not None:
        sha256 = hashlib.sha256()
    response = requests.get(url, auth=(username, password), stream=True)
    if response.status_code == 200:
        file_size = int(response.headers.get("content-length", 0))
        with open(file_path, "wb") as out:
            with tqdm(
                    total=file_size,
                    disable=not progress,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for buffer in response.iter_content():
                    out.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                )
        return True
    raise RuntimeError(
        f"Couldn't download from url {url}, got response status code {response.status_code}"
    )


def chart_dependencies(model, n_mels=80, device="cpu"):
    """
    Use backprop to chart dependencies
    (see http://karpathy.github.io/2019/04/25/recipe/)
    """
    model.eval()
    batch_size, time_steps = random.randint(2, 10), random.randint(10, 100)
    inputs = torch.randn((batch_size, n_mels, time_steps)).to(device)
    inputs.requires_grad = True
    outputs = model(inputs)
    random_index = random.randint(0, batch_size)
    loss = outputs[random_index].sum()
    loss.backward()
    assert (
               torch.cat([inputs.grad[i] == 0 for i in range(batch_size) if i != random_index])
           ).all() and (
                   inputs.grad[random_index] != 0
           ).any(), f"Only index {random_index} should have non-zero gradients"


def optimizer_to(optimizer, device="cpu"):
    """
    Transfer the given optimizer to device
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for sub_param in param.values():
                if isinstance(sub_param, torch.Tensor):
                    sub_param.data = sub_param.data.to(device)
                    if sub_param._grad is not None:
                        sub_param._grad.data = sub_param._grad.data.to(device)
    return optimizer


def scheduler_to(scheduler, device="cpu"):
    """
    Transfer the given LR scheduler to device
    """
    for param in scheduler.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
    return scheduler


def init_wandb(api_key_file, project, entity, name=None, config=None):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    print(f"my wandb key is {api_key_value}")
    return wandb.init(
        name=name,
        project=project,
        entity=entity,
        config=config,
    )


def get_train_val_metrics(y_true, y_pred, prefix=None):
    """
    Return a dictionary of classification metrics
    """
    init_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    if prefix is not None:
        init_metrics = {f"{prefix}/{k}": v for k, v in init_metrics.items()}
    return init_metrics


def get_test_metrics(
        scores, labels, mindcf_p_target=1e-2, mindcf_c_fa=1, mindcf_c_miss=1, prefix=None
):
    """
    Return EER and minDCF metrics
    """
    init_metrics = {
        "eer": compute_eer(scores, labels),
        "mindcf": compute_mindcf(
            scores,
            labels,
            p_target=mindcf_p_target,
            c_fa=mindcf_c_fa,
            c_miss=mindcf_c_miss,
        ),
    }
    if prefix is not None:
        init_metrics = {f"{prefix}/{k}": v for k, v in init_metrics.items()}
    return init_metrics


def compute_eer(scores, labels):
    """
    Compute the equal error rate score
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def compute_error_rates(scores, labels, eps=1e-6):
    """
    Creates a list of false negative rates, a list of false positive rates
    and a list of decision thresholds that give those error rates
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the error-rates are evaluated.
    sorted_indexes, _ = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=lambda t: t[1],
        )
    )
    labels = [labels[i] for i in sorted_indexes]

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i] and fprs[i]
    # is the total number of times that we have correctly accepted
    # scores greater than thresholds[i]
    fnrs, fprs = [], []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / (float(fnrs_norm) + eps) for x in fnrs]

    # Divide by the total number of correct positives to get the
    # true positive rate and subtract these quantities from 1 to
    # get the false positive rates
    fprs = [1 - x / (float(fprs_norm) + eps) for x in fprs]

    return fnrs, fprs


def compute_mindcf(scores, labels, p_target=1e-2, c_fa=1, c_miss=1, eps=1e-6):
    """
    Computes the minimum of the detection cost function
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Extract false negative and false positive rates
    fnrs, fprs = compute_error_rates(scores, labels)

    # Compute the minimum detection cost
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    # Compute default cost and use it to normalize the
    # minimum detection cost
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / (c_def + eps)

    return min_dcf


class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def visualize_embeddings(
        embeddings,
        labels,
        labels_mapping=None,
        reduction_method="umap",
        remove_outliers=False,
        only_centroids=True,
        convex_hull=False,
        figsize=(12, 10),
        legend=False,
        show=True,
        save=None,
):
    """
    Plot the given embedding vectors, after reducing them to 2D
    """
    # Convert embeddings and labels to numpy
    embeddings, labels = to_numpy(embeddings), to_numpy(labels)

    # Check inputs
    assert (
            len(embeddings.shape) == 2 and embeddings.shape[1] > 1
    ), "Wrong embeddings format/dimension"
    assert (
            len(labels.shape) == 1 and labels.shape[0] == embeddings.shape[0]
    ), "Wrong labels format/dimension"
    assert not (
            only_centroids and convex_hull
    ), "Cannot compute convex hull when only centroids are displayed"

    # Compute dimesionality reduction to 2D
    if embeddings.shape[1] > 2:
        embeddings = reduce(
            embeddings, n_components=2, reduction_method=reduction_method
        )

    # Store embeddings in a dataframe and compute cluster colors
    embeddings_df = pd.DataFrame(embeddings, columns=["x", "y"], dtype=np.float32)
    embeddings_df["l"] = np.expand_dims(labels, axis=-1)
    cluster_colors = {l: np.random.random(3) for l in np.unique(labels)}
    embeddings_df["c"] = embeddings_df.l.map(
        {l: tuple(c) for l, c in cluster_colors.items()}
    )

    # Plot embeddings and centroids
    fig, ax = plt.subplots(figsize=figsize)
    for l, c in cluster_colors.items():
        to_plot = embeddings_df[embeddings_df.l == l]
        label = labels_mapping[l] if labels_mapping is not None else l
        ax.scatter(
            to_plot.x.mean(),
            to_plot.y.mean(),
            color=c,
            label=f"{label} (C)",
            marker="^",
            s=250,
        )
        if not only_centroids:
            ax.scatter(to_plot.x, to_plot.y, color=c, label=f"{label}")

    # Do not represent outliers
    if remove_outliers:
        xmin_quantile = np.quantile(embeddings[:, 0], q=0.01)
        xmax_quantile = np.quantile(embeddings[:, 0], q=0.99)
        ymin_quantile = np.quantile(embeddings[:, 1], q=0.01)
        ymax_quantile = np.quantile(embeddings[:, 1], q=0.99)
        ax.set_xlim(xmin_quantile, xmax_quantile)
        ax.set_ylim(ymin_quantile, ymax_quantile)

    # Plot a shaded polygon around each cluster
    if convex_hull:
        for l, c in cluster_colors.items():
            try:
                # Get the convex hull
                points = embeddings_df[embeddings_df.l == l][["x", "y"]].values
                hull = ConvexHull(points)
                x_hull = np.append(
                    points[hull.vertices, 0], points[hull.vertices, 0][0]
                )
                y_hull = np.append(
                    points[hull.vertices, 1], points[hull.vertices, 1][0]
                )

                # Interpolate to get a smoother figure
                dist = np.sqrt(
                    (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
                )
                dist_along = np.concatenate(([0], dist.cumsum()))
                spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interp_x, interp_y = interpolate.splev(interp_d, spline)

                # Plot the smooth polygon
                ax.fill(interp_x, interp_y, "--", color=c, alpha=0.2)
            except:
                continue

    # Spawn the plot
    if legend:
        plt.legend()
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
    if show:
        plt.show()
    else:
        plt.close(fig)


def to_numpy(arr):
    """
    Convert the given array to the numpy format
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return None


def reduce(embeddings, n_components=2, reduction_method="umap", seed=42):
    """
    Applies the selected dimensionality reduction technique
    to the given input data
    """
    assert reduction_method in ("svd", "tsne", "umap"), "Unsupported reduction method"
    if reduction_method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=n_components, metric="cosine", random_state=seed)
    elif reduction_method == "umap":
        reducer = umap.UMAP(
            n_components=n_components, metric="cosine", random_state=seed
        )
    return reducer.fit_transform(embeddings)
