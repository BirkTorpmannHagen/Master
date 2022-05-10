import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from utils.formatting import SafeDict
from scipy.stats import wasserstein_distance
from scipy.stats import ttest_ind, pearsonr, ttest_rel
from models.segmentation_models import *


def training_plot(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("Training Plot Sample")
    plt.xlabel("Epochs")
    plt.ylabel("Jaccard Loss")
    plt.xlim((0, 300))
    plt.ylim((0, 1))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Training Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss")
    # plt.plot(log_df["epoch"], log_df["ood_iou"], label="Etis-LaribDB iou")
    plt.legend()
    plt.show()


def ood_correlations(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("SIS-OOD correlation")
    plt.xlabel("SIS")
    plt.ylabel("Etis-LaribDB OOD performance")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.scatter(log_df["consistency"], log_df["ood_iou"], label="Consistency")
    plt.scatter(log_df["iid_test_iou"], log_df["ood_iou"], label="IID IoU")

    plt.legend()
    plt.show()


def ood_v_epoch(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("Training Plot Sample")
    plt.xlabel("Epochs")
    plt.ylabel("SIL")
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.plot(log_df["epoch"], log_df["consistency"], label="consistency")
    plt.plot(log_df["epoch"], log_df["ood_iou"], label="ood iou")
    plt.legend()
    plt.show()


def modelwise_boxplot():
    results = torch.load("experiments/Data/pickles/ious_CVC-ClinicDB_DeepLab__j_0")  # vanilla


def get_boxplots_for_models():
    """
    box plot for comparing model performance. Considers d% reduced along datasets, split according to experiments
    and models
    :return:
    """
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN, Unet, InductiveNet, TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "0" in fname:
            with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
                model = fname.split("_")[0]
                if model == "InductiveNet":
                    model = "DD-DeepLabV3+"
                data = pickle.load(file)
                datasets, samples = data["ious"].shape
                kvasir_ious = data["ious"][0]
                print(kvasir_ious)
                # print(kvasir_ious)
                mean_iid_iou = np.median(kvasir_ious)
                print(mean_iid_iou)
                if "maximum_consistency" in fname:
                    continue
                for i in range(datasets):
                    # if i==0:
                    #     continue
                    for j in range(samples):
                        if data["ious"][i, j] < 0.25 or data["ious"][0][j] < 0.75:
                            print(f"{fname} with id {j} has iou {data['ious'][i, j]} and {data['ious'][0][j]} ")
                            continue
                        dataset.append([dataset_names[i], model, data["ious"][i, j]])

                        # dataset.append(
                        #     [dataset_names[i], model, 100 * (data["ious"][i, j] - mean_iid_iou) / mean_iid_iou])

    dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "\u0394%IoU"])
    sns.barplot(x="Dataset", y="\u0394%IoU", hue="Model", data=dataset)
    plt.ylim((0, 1))
    plt.show()


def get_variances_for_models():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN, Unet, InductiveNet, TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "maximum_consistency" in fname:
            continue
        if "0" in fname:
            with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
                model = fname.split("_")[0]
                if model == "InductiveNet":
                    model = "DD-DeepLabV3+"
                data = pickle.load(file)
                datasets, samples = data["ious"].shape

                if "maximum_consistency" in fname:
                    continue
                for i in range(datasets):
                    # if i == 0:
                    #     continue

                    for j in range(samples):
                        if data["ious"][0][j] < 0.75:
                            print(fname, "-", j)
                            continue
                        if i == 3 and model == "InductiveNet":
                            print("inductivenet", data["ious"][i, j])
                        if i == 3 and model == "DeepLab":
                            print("DeepLab", data["ious"][i, j])

                        dataset.append([dataset_names[i], model, data["ious"][i, j]])

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "Coefficient of Std.Dev"])
    std_dataset = iou_dataset.groupby(["Model", "Dataset"]).std() / iou_dataset.groupby(["Model", "Dataset"]).mean()
    std_dataset = std_dataset.reset_index()
    print(std_dataset)
    plt.ylim((0, 0.15))
    sns.barplot(x="Dataset", y="Coefficient of Std.Dev", hue="Model", data=std_dataset)
    plt.show()


def plot_parameters_sizes():
    models = [DeepLab, FPN, InductiveNet, Unet, TriUnet]
    model_names = ["DeepLab", "FPN", "InductiveNet", "Unet", "TriUnet"]
    for model_name, model_c in zip(model_names, models):
        model = model_c()
        print(f"{model_name}: {sum(p.numel() for p in model.parameters(recurse=True))}")


def collate_ensemble_results_into_df(type="consistency"):
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN", "Unet", "InductiveNet", "TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if type == "consistency" and "augmentation" in fname:
            continue
        if type == "augmentation" and "augmentation" not in fname:
            continue

        if "ensemble" not in fname:
            continue
        if "maximum_consistency" in fname or "last_epoch" in fname:
            continue

        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            model = fname.split("-")[0]
            experiment = fname.split("-")[-1]
            # todo fnames with consistency and augmentation
            data = pickle.load(file)
            # print(file, data.keys())
            datasets, samples = data["ious"].shape
            if model == "InductiveNet":
                model = "DD-DeepLabV3+"
            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:
                        continue

                    try:
                        dataset.append(
                            [dataset_names[i], model, j, experiment, data["ious"][i, j], data["constituents"][j]])
                    except KeyError:
                        continue

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "ID", "Experiment", "IoU", "constituents"])
    # print(iou_dataset)
    iou_dataset.to_csv("base_data.csv")
    return iou_dataset


def collate_base_results_into_df():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN", "Unet", "InductiveNet", "TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "ensemble" in fname:
            # print(fname)
            continue
        if "maximum_consistency" in fname or "last_epoch" in fname:
            # print(fname)
            continue

        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            model = fname.split("_")[0]
            data = pickle.load(file)
            datasets, samples = data["ious"].shape
            if model == "InductiveNet":
                model = "DD-DeepLabV3+"
            experiment = "No Augmentation"
            if "sil" in fname and "_G" not in fname:
                experiment = "Consistency Training"
            elif "_V" in fname:
                experiment = "Vanilla Augmentation"
            elif "_G" in fname:
                experiment = "Inpainter Augmentation"

            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:
                        continue
                    dataset.append([dataset_names[i], model, j, experiment, data["ious"][i, j], data["sis"][i, j]])

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "ID", "Experiment", "IoU", "SIS"])
    iou_dataset.to_csv("base_data.csv")
    return iou_dataset


def plot_ensemble_performance():
    df = collate_ensemble_results_into_df("augmentation")
    print(df)
    latex = df.groupby(["Model", "Dataset"])["IoU"].mean()
    print(latex.reset_index())
    print(latex.to_latex(float_format="%.3f"))
    order = df.groupby(["Dataset", "Model"])["IoU"].mean().sort_values().index
    sns.barplot(data=df, x="Dataset", y="IoU", hue="Model")
    plt.show()
    grouped_mean = df.groupby(["Dataset", "Model", "ID"])["IoU"].mean()
    # print(grouped_mean)
    grouped_iid = np.abs(grouped_mean - grouped_mean["Kvasir-SEG"]) / grouped_mean["Kvasir-SEG"]
    # print(grouped_iid)

    nedf = collate_base_results_into_df()
    nedf = nedf[nedf["Experiment"]=="Vanilla Augmentation"]
    ne_grouped_mean = nedf.groupby(["Dataset", "Model"])["IoU"].mean()
    # print(ne_grouped_mean)
    ne_grouped_iid = np.abs(ne_grouped_mean["Kvasir-SEG"] - ne_grouped_mean) / ne_grouped_mean["Kvasir-SEG"]
    # print(ne_grouped_iid)

    comparison = ne_grouped_iid - grouped_iid
    comparison = comparison.reset_index()

    sns.barplot(data=comparison, x="Dataset", y="IoU", hue="Model")
    plt.show()

    # plot delta vs variance
    ne_grouped_coeff_std = nedf.groupby(["Dataset", "Model"])["IoU"].std() / ne_grouped_mean
    ne_grouped_coeff_std = ne_grouped_coeff_std.reset_index()
    ne_grouped_coeff_std = ne_grouped_coeff_std.rename(columns={"IoU": "Coeff. StD of IoUs"})
    # print(ne_grouped_coeff_std.head(10))
    sns.barplot(data=ne_grouped_coeff_std, x="Dataset", y="Coeff. StD of IoUs", hue="Model")
    plt.show()
    test = pd.merge(ne_grouped_coeff_std, comparison)
    test = test.rename(columns={"IoU": "% Improvement over mean constituent IoU"})
    test["% Improvement over mean constituent IoU"] *= 100
    test = test.groupby(["Model", "ID"]).mean()
    test = test.reset_index()

    print("mean", np.mean(test))
    print("max", np.max(test))
    # print(test)

    sns.lineplot(data=test, x="Coeff. StD of IoUs", y="% Improvement over mean constituent IoU", err_style="bars",
                 color="gray", linestyle='--')
    test = test.groupby("Model").mean().reset_index()
    sns.scatterplot(test["Coeff. StD of IoUs"], test["% Improvement over mean constituent IoU"], hue=test["Model"],
                    s=100, ci=99)
    plt.show()


def plot_overall_ensemble_performance():
    df = collate_ensemble_results_into_df("both")
    grouped_mean = df.groupby(["Dataset", "Model", "ID"])["IoU"].mean()

    nedf = collate_base_results_into_df()
    ne_grouped_mean = nedf.groupby(["Dataset", "Model"])["IoU"].mean()

    # plot delta vs variance
    ne_grouped_coeff_std = nedf.groupby(["Dataset", "Model"])["IoU"].std() / ne_grouped_mean
    ne_grouped_coeff_std = ne_grouped_coeff_std.reset_index()
    ne_grouped_coeff_std = ne_grouped_coeff_std.rename(columns={"IoU": "Coeff. StD of IoUs"})

def plot_cons_vs_aug_ensembles():
    df = collate_ensemble_results_into_df("consistency")
    df2 = collate_ensemble_results_into_df("augmentation")
    grouped = df2.groupby(["Model", "Dataset"])["IoU"].mean()
    grouped2 = df2.groupby([ "Dataset"])["IoU"].mean()
    grouped3 = df.groupby([ "Dataset"])["IoU"].mean()

    print(grouped2)
    print(grouped3)
    latex = grouped.to_latex(float_format="%.3f")
    for dset in np.unique(df2["Dataset"]):
        print(dset)
        ttest = ttest_ind(df[df["Dataset"]==dset]["IoU"],df2[df2["Dataset"]==dset]["IoU"], equal_var=False)
        print(ttest)


def plot_inpainter_vs_conventional_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] != "Consistency Training"]

    table = df.groupby(["Dataset", "Model", "Experiment"])["IoU"].mean()
    no_augmentation = df[df["Experiment"] == "No Augmentation"].groupby(["Dataset"])[
        "IoU"].mean()

    improvements = 100 * (table - no_augmentation) / no_augmentation
    improvements = improvements.reset_index()
    improvements = improvements[improvements["Experiment"] != "No Augmentation"]
    improvements.rename(columns={"IoU": "% Change in mean IoU with respect to No Augmentation"}, inplace=True)

    test = table.to_latex(float_format="%.3f")
    print(np.max(improvements))
    print(np.mean(improvements))
    sns.boxplot(data=improvements, x="Dataset", y="% Change in mean IoU with respect to No Augmentation",
                hue="Experiment")
    plt.savefig("augmentation_plot.eps")
    return table


def plot_training_procedure_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] != "Inpainter Augmentation"]
    index = df.index[df["Experiment"] == "No Augmentation"].tolist() + df.index[
        df["Experiment"] == "Vanilla Augmentation"].tolist() + df.index[
                df["Experiment"] == "Consistency Training"].tolist()
    df = df.reindex(index)
    # print(df)
    filt = df.groupby(["Dataset", "ID", "IoU", "Experiment"]).mean()
    filt = filt.reset_index()
    hue_order = df.groupby(["Experiment"])["IoU"].mean().sort_values().index
    order = df.groupby(["Dataset"])["IoU"].mean().sort_values().index
    table = df.groupby(["Dataset", "Model", "Experiment"])["IoU"].mean()

    w_p_values = table.reset_index()
    for i, row in w_p_values.iterrows():
        experiment = row["Experiment"]
        model = row["Model"]
        dataset = row["Dataset"]
        ious = df[(df["Dataset"] == dataset) & (df["Model"] == model) & (df["Experiment"] == experiment)]["IoU"]
        augmentation_ious = \
            df[(df["Dataset"] == dataset) & (df["Model"] == model) & (df["Experiment"] == "Vanilla Augmentation")][
                "IoU"]

        w_p_values.at[i, "p-value"] = round(ttest_ind(ious, augmentation_ious, equal_var=False)[-1], 3)

    for dset in np.unique(df["Dataset"]):
        overall_ttest = ttest_ind(df[(df["Experiment"] == "Consistency Training") & (df["Dataset"] == dset)]["IoU"],
                                  df[(df["Experiment"] == "Vanilla Augmentation") & (df["Dataset"] == dset)]["IoU"],
                                  equal_var=False)
        print(f"{dset}: {overall_ttest[0]}, p={1 - round(overall_ttest[1], 5)} ")

    test = table.to_latex(float_format="%.3f")
    no_augmentation_performance = filt[filt["Experiment"] == "No Augmentation"].groupby(["Dataset"])["IoU"].mean()

    # C.StD analysis
    cstd = filt.groupby(["Dataset", "Experiment"])["IoU"].std() / filt.groupby(["Dataset", "Experiment"])[
        "IoU"].mean()
    cstd = cstd.reset_index()
    cstd.rename(columns={"IoU": "Coefficient of Standard Deviation of IoUs"}, inplace=True)
    sns.barplot(data=cstd, x="Dataset", y="Coefficient of Standard Deviation of IoUs", hue="Experiment",
                hue_order=["No Augmentation", "Vanilla Augmentation", "Consistency Training"])
    plt.savefig("consistency_training_cstd.eps")
    plt.show()
    improvement_pct = 100 * (filt.groupby(["Dataset", "Experiment", "ID"])[
                                 "IoU"].mean() - no_augmentation_performance) / no_augmentation_performance
    improvement_pct = improvement_pct.reset_index()
    print(improvement_pct[improvement_pct["Experiment"] == "No Augmentation"])
    improvement_pct = improvement_pct[improvement_pct["Experiment"] != "No Augmentation"]

    # print(np.max(improvement_pct[improvement_pct["Experiment"] == "Consistency Training"]))
    # print(np.mean(improvement_pct[improvement_pct["Experiment"] == "Consistency Training"]))
    print(np.max(improvement_pct[improvement_pct["Experiment"] == "Vanilla Augmentation"]))
    print(np.mean(improvement_pct[improvement_pct["Experiment"] == "Vanilla Augmentation"]))
    improvement_pct.rename(columns={"IoU": "% Change in mean IoU with respect to No Augmentation"}, inplace=True)
    sns.boxplot(data=improvement_pct, x="Dataset", y="% Change in mean IoU with respect to No Augmentation",
                hue="Experiment")

    plt.savefig("consistency_training_percent.eps")
    plt.show()
    # print(w_p_values)
    # scatter = sns.barplot(data=filt, x="Dataset", y="IoU", hue="Experiment", hue_order=hue_order, order=order)
    # scatter.legend(loc='lower right')
    # plt.show()
    return table


def plot_baseline_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] == "No Augmentation"]
    df_van = df.groupby(["Dataset", "Model"])["IoU"].mean()
    df_van = df_van.reset_index()
    # hue_order = df_van.groupby(["Model"])["IoU"].mean().sort_values().index
    order = df_van.groupby(["Dataset"])["IoU"].mean().sort_values().index
    print(df_van)
    # t tests here
    plt.hist(df[df["Dataset"] == "Kvasir-SEG"]["IoU"])
    plt.show()
    sns.barplot(data=df, x="Dataset", y="IoU", hue="Model", order=order)
    plt.show()


def plot_consistencies():
    df = collate_base_results_into_df()
    df.groupby(["Experiment", "Dataset", "Model", "ID"]).mean().reset_index().to_csv("test.csv")
    grouped = df.groupby(["Experiment", "Dataset", "Model", "ID"])["SIS"].mean().reset_index()
    grouped = grouped[grouped["Experiment"] != "Inpainter Augmentation"]
    grouped = grouped[grouped["Dataset"] == "Kvasir-SEG"]
    # grouped.to_csv("test.csv")
    sns.barplot(data=grouped, x="Model", y="SIS", hue="Experiment")
    plt.show()

    grouped = df.groupby(["Experiment", "Dataset", "Model", "ID"])["IoU"].mean().reset_index()
    grouped = grouped[grouped["Experiment"] != "Inpainter Augmentation"]
    grouped = grouped[grouped["Dataset"] == "Kvasir-SEG"]
    # grouped.to_csv("test.csv")
    sns.barplot(data=grouped, x="Model", y="IoU", hue="Experiment")
    plt.tight_layout()
    plt.show()

    # aug_consistencies = []
    # aug_oods = []
    # cons_consistencies = []
    # cons_oods
    cons_df = pd.DataFrame()
    aug_df = pd.DataFrame()
    for file in os.listdir("logs/consistency/FPN"):
        if "augmentation" in file:
            aug_df = aug_df.append(pd.read_csv(os.path.join("logs/consistency/FPN", file)), ignore_index=True)
        if "consistency" in file:
            cons_df = aug_df.append(pd.read_csv(os.path.join("logs/consistency/FPN", file)), ignore_index=True)
        else:
            continue
    cons_df = cons_df[cons_df["epoch"] < 300]
    aug_df = aug_df[aug_df["epoch"] < 300]
    sns.lineplot(data=cons_df, x="epoch", y="consistency", color="orange")
    sns.lineplot(data=aug_df, x="epoch", y="consistency", color="blue")
    sns.lineplot(data=cons_df, x="epoch", y="ood_iou", color="orange")
    sns.lineplot(data=aug_df, x="epoch", y="ood_iou", color="blue")
    plt.show()


def plot_ensemble_variance_relationship():
    df = collate_ensemble_results_into_df()
    df_constituents = collate_base_results_into_df()
    df["constituents"] = df["constituents"].apply(
        lambda x: [int(i.split("_")[-1]) for i in x] if type(x) == type([]) else int(x))
    df_constituents = df_constituents[df_constituents["Experiment"] == "Consistency Training"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # colors = ["b", "g", "r", "c", "m", "y"]
    colormap = dict(zip(np.unique(df["Dataset"]), colors))
    print(colormap)
    var_dataset = pd.DataFrame()
    for i, row in df.iterrows():
        model = df.at[i, "Model"]
        id = df.at[i, "ID"]
        experiment = df.at[i, "Experiment"].split(".")[0]
        if model == "diverse" and experiment != "consistency":
            continue

        if model == "diverse":
            # get non-ensemble stats
            # continue
            filtered = df_constituents[
                (df_constituents["ID"] == id) &
                (df_constituents["Experiment"] == "Consistency Training")]  # todo take augmentation into account
            cstd = (filtered.groupby(["Dataset"]).std() / filtered.groupby(["Dataset"]).mean())["IoU"]

            improvements = df[
                (df["Model"] == model) & (df["Experiment"] == f"{experiment}.pkl") & (df["ID"] == id)]

            improvements = 100 * (improvements.groupby(["Dataset"])["IoU"].mean() - filtered.groupby(["Dataset"])[
                "IoU"].mean()) / filtered.groupby(["Dataset"])["IoU"].mean()
            cstd = cstd.reset_index()
            improvements = improvements.reset_index()
            cstd.rename(columns={"IoU": "C.StD"}, inplace=True)
            improvements.rename(columns={"IoU": "% Increase in Generalizability wrt Constituents Mean"}, inplace=True)
            merged = pd.merge(improvements, cstd)
            merged["Model"] = [model] * 4
            merged["ID"] = [id] * 4
            var_dataset = var_dataset.append(merged)
        else:

            constituents = df.at[i, "constituents"]
            filtered = df_constituents[
                (df_constituents["Model"] == model) & (df_constituents["ID"].isin(constituents))]
            cstd = (filtered.groupby(["Dataset"]).std() / filtered.groupby(["Dataset"]).mean())["IoU"]
            improvements = df[
                (df["Model"] == model) & (df["Experiment"] == f"{experiment}.pkl") & (df["ID"] == id)]

            improvements = 100 * (improvements.groupby(["Dataset"])["IoU"].mean() - filtered.groupby(["Dataset"])[
                "IoU"].mean()) / filtered.groupby(["Dataset"])["IoU"].mean()
            cstd = cstd.reset_index()
            improvements = improvements.reset_index()
            cstd.rename(columns={"IoU": "C.StD"}, inplace=True)
            improvements.rename(columns={"IoU": "% Increase in Generalizability wrt Constituents Mean"}, inplace=True)
            merged = pd.merge(improvements, cstd)
            merged["Model"] = [model] * 4
            merged["ID"] = [id] * 4
            var_dataset = var_dataset.append(merged)
            # improvements = filtered.groupby
            # cstd = filtered
        # df.at[i, "cstd"] =
        # cstds.append(0)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for i, dataset_name in enumerate(np.unique(var_dataset["Dataset"])):
        dataset_filtered = var_dataset[var_dataset["Dataset"] == dataset_name]
        sns.regplot(ax=ax.flatten()[i], data=dataset_filtered, x="C.StD",
                    y="% Increase in Generalizability wrt Constituents Mean",
                    ci=99,
                    color=colormap[dataset_name], label=dataset_name)
        correlation = pearsonr(dataset_filtered["C.StD"],
                               dataset_filtered["% Increase in Generalizability wrt Constituents Mean"])
        ax.flatten()[i].set_title(f"{dataset_name}: PCC={correlation[0]:.3f}, p={correlation[1]:.6f}")
        print(dataset_name)
        print(correlation)
    for a in ax.flatten():
        a.set(xlabel=None)
        a.set(ylabel=None)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.ylabel("% Increase in Generalizability wrt Constituents Mean")
    plt.xlabel("Coefficient of Standard Deviation")
    plt.legend(labels=np.unique(var_dataset["Dataset"]))
    # plt.title()
    fig.tight_layout(pad=3)
    plt.savefig("ensemble_variance_relationship_statistical.eps")
    plt.show()
    # hue_order = var_dataset.groupby(["Model"])[
    #     "% Increase in Generalizability wrt Constituents Mean"].mean().sort_values().index
    var_dataset = var_dataset.replace("diverse", "MultiModel")
    print(var_dataset.groupby(["Dataset"]).mean())
    sns.boxplot(data=var_dataset, x="Dataset", y="% Increase in Generalizability wrt Constituents Mean", hue="Model",
                order=["Kvasir-SEG", "CVC-ClinicDB", "EndoCV2020", "Etis-LaribDB"])
    plt.axhline(0, linestyle="--")
    plt.show()
    var_dataset = var_dataset
    sns.scatterplot(data=var_dataset, x="C.StD", y="% Increase in Generalizability wrt Constituents Mean",
                    hue="Model")
    plt.show()
    # print(df)


if __name__ == '__main__':
    # plot_consistencies()
    # def test(a):
    #     return np.mean()
    # get_boxplots_for_models()
    # # collate_results_into_df()
    # get_variances_for_models()
    # plot_ensemble_performance()
    # collate_base_results_into_df()
    # # plot_parameters_sizes()
    # # training_plot("logs/vanilla/DeepLab/vanilla_1.csv")
    # plot_inpainter_vs_conventional_performance()
    # plot_training_procedure_performance()
    # plot_ensemble_performance()
    # plot_baseline_performance()
    # plot_ensemble_variance_relationship()
    plot_cons_vs_aug_ensembles()