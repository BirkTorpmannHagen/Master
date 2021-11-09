import pandas as pd
import torch


def log(fname, columns, values):
    # TODO convert to numpy for lower space requirements
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    df = df.append(dict(zip(columns, values)), ignore_index=True)
    df.to_csv(fname, index=False)


def log_iou(fname, epoch, ious: torch.Tensor):
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["epoch", "iou"])
    serialized_ious = ious.flatten().numpy()
    for iou in serialized_ious:
        df = df.append(dict(zip(["epoch", "iou"], [epoch, iou])), ignore_index=True)
    df.to_csv(fname, index=False)
