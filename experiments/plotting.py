import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from utils.formatting import SafeDict
from scipy.stats import wasserstein_distance
from scipy.stats import ttest_ind, pearsonr, mannwhitneyu, spearmanr
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
        if "ensemble" not in fname:
            continue
        if "maximum_consistency" in fname or "last_epoch" in fname:
            continue
        if type != "all":
            if type == "consistency" and ("augmentation" in fname or "vanilla" in fname):
                continue
            if type == "augmentation" and "augmentation" not in fname:
                continue
            if type == "vanilla" and "vanilla" not in fname:
                continue

        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            model = fname.split("-")[0]
            # experiment = fname.split("-")[-1]

            if "vanilla" in fname:
                experiment = "No Augmentation"
            elif "augmentation" in fname:
                experiment = "Vanilla Augmentation"
            else:
                experiment = "Consistency Training"
            data = pickle.load(file)

            # print(file, data.keys())
            datasets, samples = data["ious"].shape
            if model == "InductiveNet":
                model = "DD-DeepLabV3+"
            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:  # if bugged out; rare
                        continue
                    try:
                        dataset.append(
                            [dataset_names[i], model, j, experiment, data["ious"][i, j], data["constituents"][j]])
                    except KeyError:
                        continue

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "ID", "Experiment", "IoU", "constituents"])
    # print(iou_dataset)
    iou_dataset.to_csv("ensemble_data.csv")
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
    grouped2 = df2.groupby(["Dataset"])["IoU"].mean()
    grouped3 = df.groupby(["Dataset"])["IoU"].mean()

    print(grouped2)
    print(grouped3)
    latex = grouped.to_latex(float_format="%.3f")
    for dset in np.unique(df2["Dataset"])[::-1]:
        utest = mannwhitneyu(df[df["Dataset"] == dset]["IoU"], df2[df2["Dataset"] == dset]["IoU"])
        print(f"{dset} & {round(utest[0], 5)} & {round(utest[1], 5)} \\\ ")


def plot_inpainter_vs_conventional_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] != "Consistency Training"]
    models = np.unique(df["Model"])
    for dset in np.unique(df["Dataset"])[::-1]:
        overall_utest = mannwhitneyu(df[(df["Experiment"] == "Vanilla Augmentation") & (df["Dataset"] == dset)]["IoU"],
                                     df[(df["Experiment"] == "Inpainter Augmentation") & (df["Dataset"] == dset)][
                                         "IoU"])
        print(f"{dset} & {overall_utest[0]}, p={round(overall_utest[1], 5)} \\\ ")

    for model in models:
        print(f"{model}", end="")
        for dset in np.unique(df["Dataset"]):
            ttest = ttest_ind(
                df[(df["Experiment"] == "Inpainter Augmentation") & (df["Dataset"] == dset) & (df["Model"] == model)][
                    "IoU"],
                df[(df["Experiment"] == "Vanilla Augmentation") & (df["Dataset"] == dset) & (df["Model"] == model)][
                    "IoU"],
                equal_var=False)
            print(f" & {round(ttest[1], 5)}", end="")
        print("\\\ ")
    table = df.groupby(["Dataset", "Model", "Experiment"])["IoU"].mean()
    no_augmentation = df[df["Experiment"] == "No Augmentation"].groupby(["Dataset"])[
        "IoU"].mean()

    improvements = 100 * (table - no_augmentation) / no_augmentation
    improvements = improvements.reset_index()
    improvements = improvements[improvements["Experiment"] != "No Augmentation"]
    improvements.rename(columns={"IoU": "% Change in mean IoU with respect to No Augmentation"}, inplace=True)

    test = table.to_latex(float_format="%.3f")
    # improvements = improvements[improvements["Dataset"] == "CVC-ClinicDB"]
    print(np.max(improvements[improvements["Expe riment"] == "Vanilla Augmentation"]))
    print(np.mean(improvements[improvements["Experiment"] == "Vanilla Augmentation"]))

    print(np.max(improvements[improvements["Experiment"] == "Inpainter Augmentation"]))
    print(np.mean(improvements[improvements["Experiment"] == "Inpainter Augmentation"]))
    sns.boxplot(data=improvements, x="Dataset", y="% Change in mean IoU with respect to No Augmentation",
                hue="Experiment")

    plt.savefig("augmentation_plot.eps")
    plt.show()
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
        overall_ttest = mannwhitneyu(df[(df["Experiment"] == "Consistency Training") & (df["Dataset"] == dset)]["IoU"],
                                     df[(df["Experiment"] == "Vanilla Augmentation") & (df["Dataset"] == dset)]["IoU"])
        print(f"{dset}: {overall_ttest[0]}, p={round(overall_ttest[1], 5)} ")

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
    augmentation_performance = filt[filt["Experiment"] == "Vanilla Augmentation"].groupby(["Dataset"])["IoU"].mean()

    test = improvement_pct = 100 * (filt.groupby(["Dataset", "Experiment", "ID"])[
                                        "IoU"].mean() - augmentation_performance) / augmentation_performance
    print(test.groupby(["Experiment"]).mean())
    input()
    improvement_pct = 100 * (filt.groupby(["Dataset", "Experiment", "ID"])[
                                 "IoU"].mean() - no_augmentation_performance) / no_augmentation_performance

    improvement_pct = improvement_pct.reset_index()
    print(improvement_pct[improvement_pct["Experiment"] == "No Augmentation"])
    improvement_pct = improvement_pct[improvement_pct["Experiment"] != "No Augmentation"]

    # print(np.max(improvement_pct[improvement_pct["Experiment"] == "Consistency Training"]))
    print("Consistency")
    print(np.mean(improvement_pct[improvement_pct["Experiment"] == "Consistency Training"]))
    print("Augmentation")
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
    p_value_matrix = np.zeros((len(np.unique(df["Model"])), len(np.unique(df["Model"]))))
    models = np.unique(df["Model"])
    print()
    np.set_printoptions(precision=5, suppress=True)
    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(8, 8))
    for didx, dataset in enumerate(np.unique(df["Dataset"])):
        for i, model in enumerate(models):
            for j, compare_model in enumerate(models):
                p_value_matrix[i, j] = round(ttest_ind(df[(df["Model"] == model) & (df["Dataset"] == dataset)]["IoU"],
                                                       df[(df["Model"] == compare_model) & (df["Dataset"] == dataset)][
                                                           "IoU"],
                                                       equal_var=False)[1], 5)

        sns.heatmap(p_value_matrix, ax=ax.flatten()[didx], annot=True, xticklabels=models, yticklabels=models,
                    cbar=False)
        ax.flatten()[didx].set_title(dataset)
    plt.tight_layout()
    plt.savefig("model_pvals.eps")
    plt.show()

    df_van = df.groupby(["Dataset", "Model"])["IoU"].mean()
    df_van = df_van.reset_index()
    order = df_van.groupby(["Dataset"])["IoU"].mean().sort_values().index

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


def plot_ensemble_variance_relationship(training_method):
    df = collate_ensemble_results_into_df(training_method)
    df_constituents = collate_base_results_into_df()
    df_constituents = df_constituents[df_constituents["Experiment"] != "Inpainter Augmentation"]
    df["constituents"] = df["constituents"].apply(
        lambda x: [int(i.split("_")[-1]) for i in x] if type(x) == type([]) else int(x))
    if training_method != "all":
        if training_method == "vanilla": training_method = "No Augmentation"
        if training_method == "augmentation": training_method = "Vanilla Augmentation"
        if training_method == "consistency": training_method = "Consistency Training"
        df_constituents = df_constituents[df_constituents["Experiment"] == training_method]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # colors = ["b", "g", "r", "c", "m", "y"]
    colormap = dict(zip(np.unique(df["Dataset"]), colors))

    var_dataset = pd.DataFrame()
    for i, row in df.iterrows():
        model = df.at[i, "Model"]
        id = df.at[i, "ID"]
        experiment = df.at[i, "Experiment"]
        if model == "diverse":
            filtered = df_constituents[
                (df_constituents["ID"] == id) &
                (df_constituents["Experiment"] == experiment)]
            cstd = (filtered.groupby(["Dataset"]).std() / filtered.groupby(["Dataset"]).mean())["IoU"]
            improvements = df[
                (df["Model"] == model) & (df["Experiment"] == experiment) & (df["ID"] == id)]
            improvements = 100 * (improvements.groupby(["Dataset"])["IoU"].mean() - filtered.groupby(["Dataset"])[
                "IoU"].mean()) / filtered.groupby(["Dataset"])["IoU"].mean()
            cstd = cstd.reset_index()
            improvements = improvements.reset_index()
            cstd.rename(columns={"IoU": "C.StD"}, inplace=True)
            improvements.rename(columns={"IoU": "% Increase in Generalizability wrt Constituents Mean"}, inplace=True)
            merged = pd.merge(improvements, cstd)
            merged["Model"] = [model] * 4  # dataset length
            merged["ID"] = [id] * 4
            merged["Experiment"] = [experiment] * 4

            var_dataset = var_dataset.append(merged)
        else:

            constituents = df.at[i, "constituents"]
            filtered = df_constituents[
                (df_constituents["Model"] == model) & (df_constituents["ID"].isin(constituents)) & (
                        df_constituents["Experiment"] == experiment)]
            cstd = (filtered.groupby(["Dataset"]).std() / filtered.groupby(["Dataset"]).mean())["IoU"]
            improvements = df[
                (df["Model"] == model) & (df["Experiment"] == experiment) & (df["ID"] == id)]
            improvements = 100 * (improvements.groupby(["Dataset"])["IoU"].mean() - filtered.groupby(["Dataset"])[
                "IoU"].mean()) / filtered.groupby(["Dataset"])["IoU"].mean()
            cstd = cstd.reset_index()

            improvements = improvements.reset_index()
            cstd.rename(columns={"IoU": "C.StD"}, inplace=True)
            improvements.rename(columns={"IoU": "% Increase in Generalizability wrt Constituents Mean"}, inplace=True)
            merged = pd.merge(improvements, cstd)
            merged["Model"] = [model] * 4
            merged["ID"] = [id] * 4
            merged["Experiment"] = [experiment] * 4
            var_dataset = var_dataset.append(merged)
            # improvements = filtered.groupby
            # cstd = filtered
        # df.at[i, "cstd"] =
        # cstds.append(0)
    print(len(np.unique(var_dataset[var_dataset["Experiment"] == "Vanilla Augmentation"][
                            "% Increase in Generalizability wrt Constituents Mean"])))
    print(len(np.unique(var_dataset[var_dataset["Experiment"] == "No Augmentation"][
                            "% Increase in Generalizability wrt Constituents Mean"])))
    print(var_dataset.columns)
    datasets = np.unique(var_dataset["Dataset"])
    training_methods = ["No Augmentation", "Vanilla Augmentation", "Consistency Training"]
    fig, ax = plt.subplots(len(datasets), len(training_methods), figsize=(11, 12))
    var_dataset = var_dataset.replace("diverse", "MultiModel")

    for i, dataset_name in enumerate(datasets):
        for j, training_method in enumerate(training_methods):
            dataset_filtered = var_dataset[
                (var_dataset["Dataset"] == dataset_name) & (var_dataset["Experiment"] == training_method)]
            # sns.regplot(ax=ax.flatten()[i], data=dataset_filtered, x="C.StD",
            #             y="% Increase in Generalizability wrt Constituents Mean",
            #             ci=99,
            #             color=colormap[dataset_name], label=dataset_name)
            # correlation = pearsonr(dataset_filtered["C.StD"],
            #                        dataset_filtered["% Increase in Generalizability wrt Constituents Mean"])
            if j == 0:  # seaborn does not like global legends
                scatter = sns.scatterplot(ax=ax[i, j], data=dataset_filtered, x="C.StD",
                                          y="% Increase in Generalizability wrt Constituents Mean",
                                          ci=99, legend=False, color=colormap[dataset_name], label=dataset_name)
                ax[i, j].set_title(training_method)

            else:
                scatter = sns.scatterplot(ax=ax[i, j], data=dataset_filtered, x="C.StD",
                                          y="% Increase in Generalizability wrt Constituents Mean",
                                          ci=99, legend=False, color=colormap[dataset_name])
            correlation = spearmanr(dataset_filtered["C.StD"],
                                    dataset_filtered["% Increase in Generalizability wrt Constituents Mean"])
            ax[i, j].set_title(f"Rs={correlation[0]:.3f}, p={correlation[1]:.6f}")
    for a in ax.flatten():
        a.set(xlabel=None)
        a.set(ylabel=None)
    for axis, col in zip(ax[0], training_methods):
        axis.annotate(col, xy=(0.5, 1.5), xytext=(0, 5),
                      xycoords='axes fraction', textcoords='offset points',
                      size='xx-large', ha='center', va='baseline')
    fig.add_subplot(111, frameon=False)
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.5), ncol=2, labels=np.unique(var_dataset["Dataset"]))
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.ylabel("% Increase in Generalizability wrt Constituents Mean")
    plt.xlabel("Coefficient of Standard Deviation")
    # plt.title()
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.2)
    plt.savefig("ensemble_variance_relationship_statistical.eps")
    plt.show()
    # hue_order = var_dataset.groupby(["Model"])[
    #     "% Increase in Generalizability wrt Constituents Mean"].mean().sort_values().index
    var_dataset = var_dataset.replace("diverse", "MultiModel")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=var_dataset, ax=ax, x="Dataset", y="% Increase in Generalizability wrt Constituents Mean",
                hue="Model",
                order=["Kvasir-SEG", "CVC-ClinicDB", "EndoCV2020", "Etis-LaribDB"])

    plt.axhline(0, linestyle="--")
    plt.savefig("improvements_due_to_ensembles.eps")
    plt.show()


def get_ensemble_p_vals():
    singular = collate_base_results_into_df()
    # cross-model t-test (not used in thesis)
    print("No augmentation")
    for mix, model in enumerate(np.unique(singular["Model"])):
        print(model, end="&")
        for dix, dataset in enumerate(np.unique(singular["Dataset"])):
            single = singular[singular["Experiment"] == "No Augmentation"]
            ensemble = collate_ensemble_results_into_df(type="vanilla")
            single = single[(single["Dataset"] == dataset) & (single["Model"] == model)]
            ensemble = ensemble[(ensemble["Dataset"] == dataset) & (ensemble["Model"] == model)]
            ttest = ttest_ind(
                single["IoU"], ensemble["IoU"], equal_var=False
            )
            print(round(ttest[1], 5), end=" & ")
        print("\\\ ")
    print("Augmentation")
    for mix, model in enumerate(np.unique(singular["Model"])):
        print(model, end="&")
        for dix, dataset in enumerate(np.unique(singular["Dataset"])):
            single = singular[singular["Experiment"] == "Vanilla Augmentation"]
            ensemble = collate_ensemble_results_into_df(type="augmentation")
            single = single[(single["Dataset"] == dataset) & (single["Model"] == model)]
            ensemble = ensemble[(ensemble["Dataset"] == dataset) & (ensemble["Model"] == model)]
            ttest = ttest_ind(
                single["IoU"], ensemble["IoU"], equal_var=False
            )
            print(round(ttest[1], 5), end=" & ")
        print("\\\ ")
    print("Consistency Training")
    for mix, model in enumerate(np.unique(singular["Model"])):
        print(model, end="&")
        for dix, dataset in enumerate(np.unique(singular["Dataset"])):
            single = singular[singular["Experiment"] == "Consistency Training"]
            ensemble = collate_ensemble_results_into_df(type="consistency")
            single = single[(single["Dataset"] == dataset) & (single["Model"] == model)]
            ensemble = ensemble[(ensemble["Dataset"] == dataset) & (ensemble["Model"] == model)]
            ttest = ttest_ind(
                single["IoU"], ensemble["IoU"], equal_var=False
            )
            print(round(ttest[1], 5), end=" & ")
        print("\\\ ")

    # model-averaged
    print("When averaged across models:")
    print("No augmentation")
    experiments_long=   ["No Augmentation", "Conventional Augmentation", "Consistency Training"]
    for dix, dataset in enumerate(np.unique(singular["Dataset"])):
        single = singular[singular["Experiment"] == "No Augmentation"]
        ensemble = collate_ensemble_results_into_df(type="vanilla")
        single = single[(single["Dataset"] == dataset)]
        ensemble = ensemble[(ensemble["Dataset"] == dataset)]
        ttest = mannwhitneyu(
            single["IoU"], ensemble["IoU"]
        )
        print(round(ttest[1], 3), end=" & ")
    print("\nAugmentation")

    for dix, dataset in enumerate(np.unique(singular["Dataset"])):
        single = singular[singular["Experiment"] == "Vanilla Augmentation"]
        ensemble = collate_ensemble_results_into_df(type="augmentation")
        single = single[(single["Dataset"] == dataset)]
        ensemble = ensemble[(ensemble["Dataset"] == dataset)]
        ttest = mannwhitneyu(
            single["IoU"], ensemble["IoU"]
        )
        print(round(ttest[1], 3), end=" & ")
    print("\nConsistency Training")
    for dix, dataset in enumerate(np.unique(singular["Dataset"])):
        single = singular[singular["Experiment"] == "Consistency Training"]
        ensemble = collate_ensemble_results_into_df(type="consistency")
        single = single[(single["Dataset"] == dataset)]
        ensemble = ensemble[(ensemble["Dataset"] == dataset)]
        ttest = mannwhitneyu(
            single["IoU"], ensemble["IoU"]
        )
        print(round(ttest[1], 3), end=" & ")

    experiments = ["vanilla", "augmentation", "consistency"]
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
    for dix, dataset in enumerate(np.unique(singular["Dataset"])):
        p_values = np.zeros((len(experiments), len(experiments)))
        for i, exp1 in enumerate(experiments):
            for j, exp2 in enumerate(experiments):
                df1 = collate_ensemble_results_into_df(exp1)
                df2 = collate_ensemble_results_into_df(exp2)
                test = mannwhitneyu(df1[df1["Dataset"] == dataset]["IoU"],
                                    df2[(df2["Dataset"] == dataset)]["IoU"])
                p_values[i, j] = round(test[1], 5)
        sns.heatmap(p_values, ax=axes.flatten()[dix], annot=True, xticklabels=experiments_long,
                    yticklabels=experiments_long,
                    cbar=False)
        ax = axes.flatten()[dix].set_title(dataset)
    plt.tight_layout()
    plt.savefig("ensemble_relative_pvals.eps")
    plt.show()


def compare_ensembles():
    singular = collate_base_results_into_df()
    singular_no_augment = singular[singular["Experiment"] == "No Augmentation"].groupby(["Dataset", "ID"])[
        "IoU"].mean()
    singular_augment = singular[singular["Experiment"] == "Vanilla Augmentation"].groupby(["Dataset", "ID"])[
        "IoU"].mean()
    singular_ct = singular[singular["Experiment"] == "Consistency Training"].groupby(["Dataset", "ID"])[
        "IoU"].mean()

    no_augment = collate_ensemble_results_into_df(type="vanilla").groupby(["Dataset", "ID"])[
        "IoU"].mean()
    augment = collate_ensemble_results_into_df(type="augmentation").groupby(["Dataset", "ID"])[
        "IoU"].mean()
    consistency = collate_ensemble_results_into_df(type="consistency").groupby(["Dataset", "ID"])[
        "IoU"].mean()

    no_augment_improvements = (100 * (no_augment - singular_no_augment) / singular_no_augment).reset_index()
    augment_improvements = (100 * (augment - singular_augment) / singular_augment).reset_index()
    ct_improvements = (100 * (consistency - singular_ct) / singular_ct).reset_index()

    no_augment_improvements["Experiment"] = pd.Series(["No Augmentation"] * len(no_augment_improvements),
                                                      index=no_augment_improvements.index)
    augment_improvements["Experiment"] = pd.Series(["Conventional Augmentation"] * len(augment_improvements),
                                                   index=augment_improvements.index)
    ct_improvements["Experiment"] = pd.Series(["Consistency Training"] * len(ct_improvements),
                                              index=ct_improvements.index)
    # print("No augmentation")
    # print(no_augment_improvements)
    # print("Augmentation")
    # print(augment_improvements)
    # print("Consistency Training")
    # print(ct_improvements)
    # print(augment_improvements)
    overall_improvements = pd.concat([no_augment_improvements, augment_improvements, ct_improvements],
                                     ignore_index=True)

    experiments = np.unique(overall_improvements["Experiment"])
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
    for dix, dataset in enumerate(np.unique(overall_improvements["Dataset"])):
        p_values = np.zeros((len(experiments), len(experiments)))
        for i, exp1 in enumerate(experiments):
            for j, exp2 in enumerate(experiments):
                test = ttest_ind(overall_improvements[(overall_improvements["Dataset"] == dataset) & (
                        overall_improvements["Experiment"] == exp1)]["IoU"],
                                 overall_improvements[(overall_improvements["Dataset"] == dataset) & (
                                         overall_improvements["Experiment"] == exp2)]["IoU"], equal_var=True)
                p_values[i, j] = test[1]
        sns.heatmap(p_values, ax=axes.flatten()[dix], annot=True, xticklabels=experiments, yticklabels=experiments,
                    cbar=False)
        ax = axes.flatten()[dix].set_title(dataset)
    plt.tight_layout()
    plt.savefig("ensemble_improvement_pvals.eps")
    plt.show()

    box = sns.boxplot(data=overall_improvements, x="Experiment", y="IoU", hue="Dataset",
                      hue_order=["Kvasir-SEG", "EndoCV2020", "CVC-ClinicDB", "Etis-LaribDB"])
    box.legend(loc="upper left")
    box.set(ylabel="Improvement in IoU (%)")
    box.set(xlabel="Training Method")
    box.axhline(0, linestyle="--")
    plt.savefig("ensemble_improvements.eps")
    print("..,.")
    print(overall_improvements.groupby(["Experiment"])["IoU"].mean())
    print(overall_improvements.groupby(["Experiment"])["IoU"].max())
    plt.show()

    grouped = singular[singular["Experiment"] != "Inpainter Augmentation"].groupby(["Model", "Dataset", "Experiment"])[
        "IoU"]
    constituent_cstd = grouped.std() / grouped.mean()
    print(constituent_cstd)


def test():
    ensemble = collate_ensemble_results_into_df("all")
    ensemble = ensemble.replace("augmentation", "Vanilla Augmentation")
    ensemble = ensemble.replace("vanilla", "No Augmentation")
    ensemble = ensemble.replace("consistency", "Consistency Training")

    ensemble = ensemble[ensemble["Model"] != "diverse"]
    ensemble_means = ensemble.groupby(["Experiment", "Dataset", "Model", "ID"])["IoU"].mean()
    singular = collate_base_results_into_df()
    singular = singular[singular["Experiment"] != "Inpainter Augmentation"]
    singular_grouped = singular.groupby(["Experiment", "Dataset", "Model"])["IoU"]
    # input()

    ensemble_improvements = 100 * (ensemble_means - singular_grouped.mean()) / singular_grouped.mean()
    singular_cstds = singular_grouped.std() / singular_grouped.mean()
    merged = pd.merge(ensemble_improvements, singular_cstds, how='inner', on=["Experiment", "Dataset", "Model"])
    # merged = merged.groupby(["Experiment", "Model"]).mean()
    fig = sns.scatterplot(data=merged, x="IoU_y", y="IoU_x", hue="Experiment")
    test = spearmanr(merged["IoU_y"], merged["IoU_x"])
    plt.title(f"R_s = {round(test[0], 5)}, p={round(test[1], 5)}")
    fig.set_ylabel("Change in IoU (%)")
    fig.set_xlabel("IoU C.StD.")
    # print(spearmanr(merged["IoU_y"], merged["IoU_x"]))

    plt.savefig("ensembles_underspecification.eps")
    plt.show()


if __name__ == '__main__':
    # plot_consistencies()
    # def test(a):
    #     return np.mean()
    # get_boxplots_for_models()
    # # collate_results_into_df()
    # get_variances_for_models()
    # plot_ensemble_performance()
    # collate_base_results_into_df()
    # plot_parameters_sizes()
    # # training_plot("logs/vanilla/DeepLab/vanilla_1.csv")
    # plot_inpainter_vs_conventional_performance()
    # plot_training_procedure_performance()
    # plot_ensemble_performance()
    # plot_baseline_performance()
    # plot_ensemble_variance_relationship("all")
    # plot_cons_vs_aug_ensembles()
    compare_ensembles()
    # get_ensemble_p_vals()
    # test()
