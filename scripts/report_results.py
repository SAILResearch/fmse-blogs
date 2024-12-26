import ast
import sys
from pathlib import Path

CUR_DIR = Path(__file__).parent.resolve()
added_path = CUR_DIR.parent.resolve()
print(added_path)
sys.path.insert(0, str(added_path))

from src import apimodels
from src import common_path
import numpy as np
import pandas as pd
from src.jury import standardize_confidence_scores, RandomForestMerger, MajorityMerger, DecisionTreeMerger
from sklearn.metrics import cohen_kappa_score

FIG_DIR = common_path.FIG_PATH
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = common_path.OUT_PATH
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

            # print(f"=== Ref Model: {ref_model}, Compare with: {compare_with} ===")
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
            # print(merged_df["category_df1"].unique(), merged_df["category_df2"].unique())
            # helper.printValueCountsPercentage(merged_df["category_df1"] == merged_df["category_df2"])

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

    all_model_categories, categories_record_dfs, model_categories_set = read_llm_output(label_root_dir, LLM_JURIES,
                                                                                        prompt_name)
    model_categories_set = list(model_categories_set)
    human_df = pd.read_csv(human_df_path)
    # helper.printValueCountsPercentage(human_df[human_df_col_name])
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
        # uncomment to save the visualization
        # if merger_type == "decision_tree":
        #     merger.save_visualization(FIG_DIR / f"DecisionTree_{prompt_name}.png")
    else:
        raise ValueError(f"Unknown merger type: {merger_type}")

    all_model_categories['voted_category'] = all_model_categories.drop(columns=["id"]).apply(
        merger.merge,
        axis=1,
        model_list=LLM_JURIES,
    )
    categories_record_dfs["voted"] = all_model_categories[["id", "voted_category"]].copy()
    categories_record_dfs["voted"].columns = ["id", "category"]
    # helper.printValueCountsPercentage(categories_record_dfs["voted"]["category"])
    categories_record_dfs["voted"].to_csv(save_path, index=False)

    results, valid_ids = calculate_all_agreements(
        LLM_JURIES + ["voted", "human"],
        categories_record_dfs
    )
    results_df = pd.DataFrame(results)
    results_df["Cohen Kappa"] = results_df["Cohen Kappa"].apply(lambda x: round(x, 2))
    print(f"Agreement Results for {prompt_name}:")
    print(results_df[results_df["Compare With"] == "human"])


def reproduce_table(csv_file):
    df = pd.read_csv(csv_file)
    # Step 1: Count unique companies by activity
    activity_count = df.groupby('activity')['company'].nunique().reset_index(name='activity_unique_companies')

    # Step 2: Explode tasks to have one task per row
    df['tasks'] = df['tasks'].apply(ast.literal_eval)
    df_exploded = df.explode('tasks')
    mask = ((df_exploded['tasks'] == "5. Model Deployment")
            | (df_exploded['tasks'] == "6. System Architecture")
            | (df_exploded['tasks'] == "3. Model Fine-Tuning")
            | (df_exploded['tasks'] == "2. Data Management"))
    df_exploded = df_exploded[~mask]

    # Step 3: Count unique companies mentioning each task, within each activity
    task_count = df_exploded.groupby(
        ['activity', 'tasks']
    )['company'].nunique().reset_index(name='task_unique_companies')

    # Step 4: Count the total occurrences of each activity and each task
    # For categories: count total rows per activity
    activity_occurrences = df.groupby('activity').size().reset_index(name='activity_occurrences')

    # For activities: count total rows per task
    task_occurrences = df_exploded.groupby(['activity', 'tasks']).size().reset_index(
        name='task_occurrences')

    # Step 5: Merge activity and task counts into one DataFrame
    merged_df = pd.merge(task_count, activity_count, on='activity', how='left')
    merged_df = pd.merge(merged_df, task_occurrences, on=['activity', 'tasks'], how='left')
    merged_df = pd.merge(merged_df, activity_occurrences, on='activity', how='left')

    # Step 6: Rearrange columns and clean up the labels
    merged_df["activity"] = merged_df["activity"].apply(lambda x: x[x.rfind(". ") + 2:])
    merged_df["task"] = merged_df["tasks"].apply(lambda x: x[x.rfind(". ") + 2:])

    # Step 7: Sort values as per the user's request
    final_df = merged_df[
        ['activity', 'activity_occurrences', 'activity_unique_companies', 'task', 'task_occurrences',
         'task_unique_companies']]
    final_df = final_df.sort_values(by=['activity_unique_companies', 'activity', 'task_unique_companies'],
                                    ascending=False)

    # Step 8: Convert DataFrame to LaTeX table
    latex_table = final_df.to_latex(index=False)

    # Print the LaTeX table
    print(latex_table)


def main():
    print("===== Reproducing results for Table 2 in the paper =====")
    # You can use different merger types: "majority", "random_forest", "decision_tree"
    # merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="random_forest")
    # merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="decision_tree")
    merge_categories(prompt_name="SEFM_Area", human_df_col_name="area", merger_type="majority")

    # merge labels for FM4SE
    merge_categories(prompt_name="FM4SE", human_df_col_name="category", merger_type="majority")

    # merge labels for SE4FM
    merge_categories(prompt_name="SE4FM", human_df_col_name="category", merger_type="majority")

    print("\n===== Reproducing Table 3 in the paper=====")
    reproduce_table(csv_file=common_path.DATA_PATH / "FM4SE_activities.csv")

    print("\n===== Reproducing Table 4 in the paper=====")
    reproduce_table(csv_file=common_path.DATA_PATH / "SE4FM_activities.csv")


if __name__ == '__main__':
    main()
