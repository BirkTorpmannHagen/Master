import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def get_best_epoch_from_id_split(df):
    mean_ious_per_epoch = [np.mean(df[df["epoch"] == i]["iou"]) for i in np.unique(df["epoch"])]
    return np.argmax(mean_ious_per_epoch)


def plot_std_iou_correlation(df, best=True):
    iou_std = []
    iou_avg = []
    for i in np.unique(df["id"]):
        id_df = df[df["id"] == i]
        id_best_epoch = get_best_epoch_from_id_split(id_df)
        if best:
            ious = id_df[id_df["epoch"] == id_best_epoch]["iou"]
            iou_std.append(np.std(ious))
            iou_avg.append(np.mean(ious))
        else:
            for e in np.unique(id_df["epoch"]):
                ious = id_df[id_df["epoch"] == e]["iou"]
                iou_std.append(np.std(ious))
                iou_avg.append(np.mean(ious))

    plt.scatter(iou_std, iou_avg)
    m, c, r, p, se = linregress(iou_std, iou_avg)
    x = np.linspace(np.min(iou_std), np.max(iou_std), 100)
    plt.plot(x, m * x + c, label="R = {}".format(r))
    plt.legend()
    plt.ylabel("Mean of IoUs")
    plt.xlabel("StDev of IoUs")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("logs/ious.log")
    plot_std_iou_correlation(df, best=False)
    # ious = df["iou"]
    # stds = []
    # mean_ious = []
    # for i in np.unique(df["id"]):
    #     plt.hist(ious[df["id"] == i][df["epoch"] == 175], alpha=0.5, bins=np.linspace(0, 1, 25), label=i, density=True)
    #     mean_ious.append(np.mean(ious[df["id"] == i][df["epoch"] == 175]))
    #     stds.append(np.std(ious[df["id"] == i][df["epoch"] == 175]))
    # # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # print(stds)
    # plt.hist(stds)
    # plt.show()
    #
    # plt.scatter(stds, mean_ious)
    # m, c, r, p, se = linregress(stds, mean_ious)
    # print(m, c)
    # plt.xlabel("Standard deviation")
    # plt.ylabel("IoU")
    # plt.show()
