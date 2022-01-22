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


def log_full(epoch, id, config, result_dict, type):
    data = {**config, **result_dict}

    try:
        df = pd.read_csv(f"logs/{type}/{config['model']}/{id}.csv")
    except FileNotFoundError:
        print("File not found, creating new")
        df = pd.DataFrame(columns=data)
    data["epoch"] = epoch

    # data now contains all scores for every sample, so iterate over samples
    new_df = df.append(data, ignore_index=True)
    new_df.to_csv(f"logs/{type}/{config['model']}/{id}.csv", index=False)
