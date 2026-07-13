from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_logistic_baseline(
    frame: pd.DataFrame,
    target: str,
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int = 42,
) -> dict[str, object]:
    model_frame = frame[categorical_features + numeric_features + [target]].dropna(subset=[target]).copy()
    y = model_frame[target].astype(int)
    x = model_frame[categorical_features + numeric_features]

    if y.nunique() < 2:
        raise ValueError(f"Target {target} has fewer than 2 classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_score = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    importance = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs, "abs_coefficient": abs(coefs)})
        .sort_values("abs_coefficient", ascending=False)
        .head(20)
    )

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "positive_rate": float(y.mean()),
        "test_rows": int(len(y_test)),
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "importance": importance,
        "scored_test": x_test.assign(actual=y_test.values, score=y_score, predicted=y_pred),
    }
