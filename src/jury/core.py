__all__ = [
    "standardize_confidence_scores",
    "MajorityMerger",
    "TreeBasedMerger",
    "RandomForestMerger",
    "DecisionTreeMerger",
]

from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz, DecisionTreeClassifier

RANDOM_STATE = 42


def standardize_confidence_scores(confidences):
    # Compute mean and standard deviation for each LLM
    means = np.mean(confidences)
    std_devs = np.std(confidences)
    # print(means, std_devs)

    # Standardize using Z-score: (x - mean) / std
    standardized_scores = (confidences - means) / std_devs
    return standardized_scores


class Merger(object):
    def __init__(self):
        pass

    def merge(self, row, model_list):
        pass


class MajorityMerger(Merger):
    def __init__(self):
        pass

    def merge(self, row, model_list):
        categories = []
        confidences = []

        # Collect categories and confidences dynamically based on the number of models
        for model in model_list:
            categories.append(row[f'category_{model.value}'])
            confidences.append(row[f'confidence_{model.value}'])

        # Count occurrences of each category
        category_counts = Counter(categories)
        most_common = category_counts.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            # No tie, return the category with most votes
            return most_common[0][0]
        else:
            # Tie situation, break by highest average confidence
            tie_categories = [item[0] for item in most_common if item[1] == most_common[0][1]]

            # Compute average confidence for each tied category
            avg_confidences = {
                category: sum([confidences[i] for i in range(len(model_list)) if categories[i] == category]) /
                          category_counts[category]
                for category in tie_categories
            }

            # Return the category with the highest average confidence
            return max(avg_confidences, key=avg_confidences.get)


class TreeBasedMerger(Merger):
    def __init__(self):
        self.clf = None  # This will hold the trained model
        self.le = None  # LabelEncoder to encode and decode labels
        self.feature_names = None

    def _init_classifier(self):
        raise NotImplementedError

    def train(self, human_df, human_df_col_name, model_list, model_categories_set):
        # Initialize the label encoder
        self.le = LabelEncoder()

        # Remove rows with NaN values in the target column
        human_df = human_df[~human_df[human_df_col_name].isna()]

        # Ground truth labels
        y = human_df[human_df_col_name]

        self.le.fit(list(model_categories_set))  # Fit the label encoder on ground truth labels
        y = self.le.transform(y)
        print("Classes:", self.le.classes_)

        # Features (LLM categories and confidence scores)
        selected_columns = []
        for model in model_list:
            selected_columns.append(f"category_{model.value}")
            selected_columns.append(f"confidence_{model.value}")
            human_df[f"category_{model.value}"] = self.le.transform(human_df[f"category_{model.value}"])
            human_df[f"confidence_{model.value}"] = human_df[f"confidence_{model.value}"].astype(float)

        X = human_df[selected_columns]
        self.feature_names = X.columns

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

        # Initialize the classifier
        self._init_classifier()

        # Train the classifier
        self.clf.fit(X_train, y_train)

        # Evaluate the model on test set
        y_pred = self.clf.predict(X_test)

        ck_value = cohen_kappa_score(y_test, y_pred)
        print(y_test, "vs.", y_pred)
        print("Agreement in the test set:", ck_value)
        print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report (Test):\n", classification_report(y_test, y_pred))

        # Evaluate on training set
        y_train_pred = self.clf.predict(X_train)
        print("Training Set Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Classification Report (Train):\n", classification_report(y_train, y_train_pred))

        # Finally, fit on all the data
        self.clf.fit(X, y)

    def merge(self, row, model_list):
        # Collect category and confidence features from the row
        features = []
        selected_columns = []  # To store the column names
        for model in model_list:
            selected_columns.append(f"category_{model.value}")
            selected_columns.append(f"confidence_{model.value}")
            # print(model)
            # print(row[f'category_{model.value}'])
            # print(self.le.transform([row[f'category_{model.value}']]))
            features.extend(self.le.transform([row[f'category_{model.value}']]))
            features.append(row[f'confidence_{model.value}'])
            # Store column names as they were used in training

        # Reshape the features for prediction (1 sample, n features)
        features = pd.DataFrame([features], columns=selected_columns)

        # Predict the category using the trained model
        prediction = self.clf.predict(features)

        # Inverse transform to get the original label
        return self.le.inverse_transform(prediction)[0]

    def save_model(self, model_path, encoder_path):
        """Save the trained RandomForest model and LabelEncoder to files."""
        if self.clf is not None and self.le is not None:
            # Save the model
            joblib.dump(self.clf, model_path)
            print(f"Model saved to {model_path}")

            # Save the LabelEncoder
            joblib.dump(self.le, encoder_path)
            print(f"Label encoder saved to {encoder_path}")
        else:
            print("Model and LabelEncoder must be trained before saving.")

    def load_model(self, model_path, encoder_path):
        # Load the model
        self.clf = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Load the LabelEncoder
        self.le = joblib.load(encoder_path)
        print(f"Label encoder loaded from {encoder_path}")


class RandomForestMerger(TreeBasedMerger):
    def _init_classifier(self):
        self.clf = RandomForestClassifier(
            min_samples_leaf=2, random_state=RANDOM_STATE, bootstrap=True
        )


class DecisionTreeMerger(TreeBasedMerger):
    def _init_classifier(self):
        self.clf = DecisionTreeClassifier(min_samples_leaf=2)

    def save_visualization(self, fig_path):
        dot_data = export_graphviz(
            self.clf,
            out_file=None,
            feature_names=self.feature_names,
            class_names=self.le.classes_,  # Assuming you've used label_encoder for string labels
            filled=True, rounded=True,
            special_characters=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        fig_path_str = str(fig_path)
        if fig_path_str.endswith("png"):
            graph.write_png(fig_path_str)
        elif fig_path_str.endswith("pdf"):
            graph.write_pdf(fig_path_str)
        else:
            print("Invalid file format. Use either PNG or PDF.")
        return graph
