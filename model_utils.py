import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        """
        docstring here
            :param self:
            :param X:
            :param y=None:
        """
        return self

    def transform(self, X, y=None):
        return X[self.column_names]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class ModelMaker:
    def __init__(
        self,
        categorical_vars=None,
        numerical_vars=None,
        model=None,
        params=None,
        imputation_numeric=None,
        scale_vars=False,
    ):

        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        if not model:
            self.model = LogisticRegression
        else:
            self.model = model

        if not params:
            self.params = {}
        else:
            self.params = params

        if imputation_numeric == None:
            self.imputation_numeric = "median"

        elif imputation_numeric not in ["mean", "median"]:
            raise ValueError("Please impute with either with 'mean' or 'median'.")

        else:
            self.imputation_numeric = imputation_numeric

        self.scale_vars = scale_vars

    def _make_pipeline(self):
        """
        docstring here
            :param self:
        """
        if self.categorical_vars:
            pipeline_categorical = Pipeline(
                [
                    ("categorical_selector", ColumnSelector(self.categorical_vars)),
                    ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    ),
                ]
            )

        if bool(self.numerical_vars) & bool(self.scale_vars):
            pipeline_numeric = Pipeline(
                [
                    ("numeric_selector", ColumnSelector(self.numerical_vars)),
                    ("scaling", StandardScaler()),
                    (
                        "numeric_imputer",
                        SimpleImputer(strategy=self.imputation_numeric),
                    ),
                ]
            )
        elif self.numerical_vars:
            pipeline_numeric = Pipeline(
                [
                    ("numeric_selector", ColumnSelector(self.numerical_vars)),
                    (
                        "numeric_imputer",
                        SimpleImputer(strategy=self.imputation_numeric),
                    ),
                ]
            )

        if bool(self.categorical_vars) & bool(self.numerical_vars):
            pipeline_preprocessing = FeatureUnion(
                [
                    ("categorical_pipeline", pipeline_categorical),
                    ("numeric_pipeline", pipeline_numeric),
                ]
            )
        else:
            pipeline_preprocessing = pipeline_numeric or pipeline_categorical

        self.pipeline_model = Pipeline(
            [
                ("preprocessing", pipeline_preprocessing),
                ("model", self.model(**self.params)),
            ]
        )
        return self

    def get_variable_names(self, X):
        """
        docstring here
            :param self:
            :param X:
        """

        cat_vars = []
        if self.categorical_vars:
            for var in self.categorical_vars:
                names = ["is_" + name for name in set(X[var])]
                cat_vars = cat_vars + names
        self.variable_names = cat_vars + self.numerical_vars

    def get_model(self):
        return self.pipeline_model.named_steps["model"]

    def get_coeffitients(self):
        try:
            m = self.get_model()
        except AttributeError:
            raise AttributeError("Fit the model first!")

        if isinstance(m, LogisticRegression):
            coeff = m.coef_.ravel()
        else:
            raise ValueError("Model is not an instance of Logistic Regression!")

        return pd.DataFrame(
            {"var_name": self.variable_names, "coef": coeff}
        ).sort_values("coef", ascending=False)

    def fit(self, X, y):
        self._make_pipeline()
        self.pipeline_model = self.pipeline_model.fit(X, y)
        self.get_variable_names(X)
        return self

    def predict(self, X, y=None):
        try:
            return self.pipeline_model.predict(X)
        except AttributeError:
            raise (AttributeError("Model is not fitted. Please fit the model first."))

    def predict_proba(self, X, y=None):
        try:
            return self.pipeline_model.predict_proba(X)
        except AttributeError:
            raise (AttributeError("Model is not fitted. Please fit the model first."))

    def visualize_precision_recall(self, X_test_, y_test_):
        if self.pipeline_model:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.axhline(0.5, color="k", ls="--", label="Random guess performance")

            precision, recall, thresholds = precision_recall_curve(
                y_test_, self.pipeline_model.predict_proba(X_test_)[:, 1]
            )
            ax.fill_between(recall, 0, precision, alpha=0.2)
            ax.plot(recall, precision, label="Model performance", color="r", alpha=0.5)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall curve of the model")
            plt.legend()
        else:
            raise ValueError("Please fit the model first!")

    def visualize_confusion_matrix(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Model accuracy is: {100*accuracy:.2f}%.")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plot_confusion_matrix(cm, figsize=(15, 8))
        ax.set_title("Confusion matrix for our classification")

    def get_feature_importance(self, plot=True):
        model = self.get_model()

        if isinstance(model, DecisionTreeClassifier) | isinstance(model, RandomForestClassifier):
            importances = self.pipeline_model.named_steps["model"].feature_importances_
        else:
            raise ValueError("Linear model has no feature importance attribute.")

        out = pd.DataFrame(
            {"feature": self.variable_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(15, 8))
        out.set_index("feature").plot(kind="bar", ax=ax, ec='k')
        return out
