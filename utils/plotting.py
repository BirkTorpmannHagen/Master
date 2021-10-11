import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, iqr


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
            iou_std.append(np.nanstd(id_df[id_df["iou"] > 0.01]["iou"]))
            iou_avg.append(np.nanmean(id_df[id_df["iou"] > 0.01]["iou"]))

    plt.scatter(iou_std, iou_avg)
    print(iou_avg)
    result = linregress(iou_std, iou_avg)
    m, c, r, p, se = result
    print(result)
    x = np.linspace(np.min(iou_std), np.max(iou_std), 100)
    plt.plot(x, m * x + c, label="R = {}".format(r))
    plt.legend()
    plt.ylabel("Mean of IoUs")
    plt.xlabel("StDev of IoUs")
    plt.show()


def compare_variances(baseline_df, stresstest_df):
    baseline_mean_ious = []
    stresstest_mean_ious = []
    for id in np.unique(stresstest_df["id"]):
        baseline_ious = baseline_df[baseline_df["id"] == id]["iou"]
        stresstest_ious = stresstest_df[stresstest_df["id"] == id]["iou"]
        baseline_mean_ious.append(np.mean(baseline_ious))
        stresstest_mean_ious.append(np.mean(stresstest_ious))
    plt.hist(baseline_mean_ious, alpha=0.5, label=f"Baseline; iqr={iqr(baseline_mean_ious)}")
    plt.hist(stresstest_mean_ious, alpha=0.5, label=f"Stresstest; iqr={iqr(stresstest_mean_ious)}")
    plt.legend()
    plt.xlabel("Mean IoU")
    plt.ylabel("P(Mean IoU)")

    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv("logs/ious.log")
    # df_untrained = pd.read_csv("logs/no-pretrain-ious.log")
    df_etis = pd.read_csv("logs/etis-ious.log")
    df_etis_stresstest = pd.read_csv("logs/etis-stresstest-results.log")
    df_kvasir_stresstest = pd.read_csv("logs/stresstest-results.log")
    df_kvasir = pd.read_csv("logs/kvasir-baseline.log")
    df_untrained = pd.read_csv("logs/kvasir-no-pretrain-baseline.log")
    # plot_std_iou_correlation(df, best=True)
    # plot_std_iou_correlation(df_untrained, best=True)
    # plot_std_iou_correlation(df_stresstest, best=False)
    # compare_variances(df_etis, df_etis_stresstest)
    # compare_variances(df, df_kvasir_stresstest)
    compare_variances(df_kvasir, df_untrained)
    plt.show()

    plot_std_iou_correlation(df_untrained, best=False)
    plot_std_iou_correlation(df, best=False)

    # compare_variances(df, df)
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
