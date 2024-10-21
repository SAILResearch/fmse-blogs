import pathlib
import sys
from pathlib import Path

added_path = Path("./").parent.resolve()
print(added_path)
sys.path.insert(0, str(added_path))

from src import apimodels
from src import helper
import numpy as np
import pandas as pd
from jury import standardize_confidence_scores, RandomForestMerger, MajorityMerger, DecisionTreeMerger
from sklearn.metrics import cohen_kappa_score

FIG_DIR = pathlib.Path("../figs")
OUTPUT_DIR = pathlib.Path("../output")
RANDOM_STATE = 3
LLM_JURIES = [
    apimodels.Models.GEMINI_1_5_FLASH_VERTEX,
    apimodels.Models.GPT_4O_MINI,
    apimodels.Models.QWEN_2_72B,
]


def compute_agreement(df1, df2):
    merged_df = pd.merge(df1, df2, on='id', suffixes=('_df1', '_df2'))
    ck_value = cohen_kappa_score(merged_df['category_df1'], merged_df['category_df2'])
    return ck_value

def calculate_all_agreements(model_list, categories_record_dfs, exclude_others=False):
    results = []
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            ref_model = model_list[i]
            compare_with = model_list[j]
            if ref_model == compare_with:
                continue

            print(f"=== Ref Model: {ref_model}, Compare with: {compare_with} ===")
            df1 = categories_record_dfs[ref_model].copy()
            df2 = categories_record_dfs[compare_with].copy()
            df1["category"] = df1["category"].apply(lambda x: x if x == "Invalid" else x)
            df2["category"] = df2["category"].apply(lambda x: x if x == "Invalid" else x)

            # Get valid IDs present in both models
            if exclude_others:
                df1 = df1[df1["category"] != "Others"].copy()
                df2 = df2[df2["category"] != "Others"].copy()
            valid_ids = set(df1["id"].unique()).intersection(set(df2["id"].unique()))
            df1 = df1[df1["id"].isin(valid_ids)]
            df2 = df2[df2["id"].isin(valid_ids)]

            # Merge the data for comparison
            merged_df = pd.merge(df1, df2, on='id', suffixes=('_df1', '_df2'))
            num_sample = len(merged_df)

            # Print unique categories for both models for debugging
            print(merged_df["category_df1"].unique(), merged_df["category_df2"].unique())
            helper.printValueCountsPercentage(merged_df["category_df1"] == merged_df["category_df2"])

            # Calculate Cohen's Kappa agreement
            kappa_category = compute_agreement(df1, df2)
            results.append({
                "Ref Model": ref_model.value if isinstance(ref_model, apimodels.Models) else ref_model,
                "Compare With": compare_with.value if isinstance(compare_with, apimodels.Models) else compare_with,
                "Cohen Kappa": kappa_category,
                "Num Samples": num_sample,
            })
    return pd.DataFrame(results), valid_ids

def read_llm_output(label_root_dir, model_list, standardize_confidence=True):
    model_categories_set = set()
    categories_record_dfs = {}
    all_model_categories = None
    for model in model_list:
        # Load the categories for the current model
        categories_record_dfs[model] = pd.read_csv(
            label_root_dir / f"{model.value}/categories_record.csv")
        model_categories_set = model_categories_set.union(set(categories_record_dfs[model]["category"].unique()))

        # Rename the category column to be unique for the current model
        categories_record_dfs[model][f"category_{model.value}"] = categories_record_dfs[model]["category"]
        if standardize_confidence:
            # Standardize confidence scores for each model
            confidences = np.array(categories_record_dfs[model]["confidence"]).reshape(-1, 1)
            standardized_scores = standardize_confidence_scores(confidences)
            categories_record_dfs[model]["confidence"] = standardized_scores
        categories_record_dfs[model][f"confidence_{model.value}"] = categories_record_dfs[model]["confidence"]

        # Merge this model's categories with the previous models
        if all_model_categories is None:
            all_model_categories = categories_record_dfs[model][
                ["id", f"category_{model.value}", f"confidence_{model.value}"]].copy()
        else:
            all_model_categories = all_model_categories.merge(
                categories_record_dfs[model][["id", f"category_{model.value}", f"confidence_{model.value}"]],
                on="id",
                how="left"
            )
    return all_model_categories, categories_record_dfs, model_categories_set


def merge_categories(prompt_name, human_df_col_name, merger_type="majority"):
    label_root_dir = OUTPUT_DIR / prompt_name
    human_df_path = label_root_dir / "human_df.csv"
    save_path = label_root_dir / "voted.csv"

    all_model_categories, categories_record_dfs, model_categories_set = read_llm_output(label_root_dir, LLM_JURIES, prompt_name)
    model_categories_set = list(model_categories_set)
    human_df = pd.read_csv(human_df_path)
    helper.printValueCountsPercentage(human_df[human_df_col_name])
    categories_record_dfs["human"] = human_df[["id", human_df_col_name]].copy()
    categories_record_dfs["human"].columns = ["id", "category"]

    if merger_type == "majority":
        merger = MajorityMerger()
    elif merger_type in ["random_forest", "decision_tree"]:
        if merger_type == "random_forest":
            merger = RandomForestMerger()
        else:
            merger = DecisionTreeMerger()
        training_df = human_df.copy()
        for model in LLM_JURIES:
            training_df = training_df.merge(
                categories_record_dfs[model][["id", f"category_{model.value}", f"confidence_{model.value}"]],
                on="id",
                how="left"
            )
        merger.train(training_df, human_df_col_name, LLM_JURIES, model_categories_set)
        if merger_type == "decision_tree":
            merger.save_visualization(FIG_DIR / f"DecisionTree_{prompt_name}.png")
    else:
        raise ValueError(f"Unknown merger type: {merger_type}")

    all_model_categories['voted_category'] = all_model_categories.drop(columns=["id"]).apply(
        merger.merge,
        axis=1,
        model_list=LLM_JURIES,
    )
    categories_record_dfs["voted"] = all_model_categories[["id", "voted_category"]].copy()
    categories_record_dfs["voted"].columns = ["id", "category"]
    helper.printValueCountsPercentage(categories_record_dfs["voted"]["category"])
    categories_record_dfs["voted"].to_csv(save_path, index=False)

    results, valid_ids = calculate_all_agreements(
        LLM_JURIES + ["voted", "human"],
        categories_record_dfs
    )
    results_df = pd.DataFrame(results)
    print("=== Agreement Results ===")
    print(results_df)


def main():
    # You can use different merger types: "majority", "random_forest", "decision_tree"
    merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="random_forest")
    merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="decision_tree")
    merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="majority")

    # merge labels for FM4SE
    merge_categories(prompt_name="FM4SE", human_df_col_name="category", merger_type="majority")

    # merge labels for SE4FM
    merge_categories(prompt_name="SE4FM", human_df_col_name="category", merger_type="majority")


if __name__ == '__main__':
    main()
