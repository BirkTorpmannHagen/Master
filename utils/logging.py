import pandas as pd
import torch


def log(fname, columns, values):
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    df = df.append(dict(zip(columns, values)), ignore_index=True)
    df.to_csv(fname, index=False)


def log_iou(fname, epoch, id, ious: torch.Tensor):
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["epoch", "id", "iou"])
    serialized_ious = ious.flatten().numpy()
    for iou in serialized_ious:
        df = df.append(dict(zip(["epoch", "id", "iou"], [epoch, id, iou])), ignore_index=True)
    df.to_csv(fname, index=False)
