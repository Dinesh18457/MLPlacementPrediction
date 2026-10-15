import base64
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

# Flask/server-friendly matplotlib backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# PREPROCESSING & ENCODING UTILITIES
# ============================================================

def normalize(df):

    num_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",
        "Internships",
    ]

    num_cols = [
        c for c in num_cols
        if c in df.columns
    ]

    stratify_column = None

    if "PlacementStatus" in df.columns:
        stratify_column = df["PlacementStatus"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_column,
    )

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_df[num_cols]
    )

    test_scaled = scaler.transform(
        test_df[num_cols]
    )

    train_df = pd.DataFrame(
        train_scaled,
        columns=num_cols,
        index=train_df.index,
    )

    test_df = pd.DataFrame(
        test_scaled,
        columns=num_cols,
        index=test_df.index,
    )

    return train_df, test_df


def normalization(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    columns = train_df.columns.tolist()

    train_df = pd.DataFrame(
        train_scaled,
        columns=columns,
        index=train_df.index,
    )

    test_df = pd.DataFrame(
        test_scaled,
        columns=columns,
        index=test_df.index,
    )

    return train_df, test_df


def standardization(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    num_cols = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops",
    ]

    num_cols = [
        c for c in num_cols
        if c in df.columns
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        train_df[num_cols]
    )

    X_test_scaled = scaler.transform(
        test_df[num_cols]
    )

    train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=num_cols,
        index=train_df.index,
    )

    test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=num_cols,
        index=test_df.index,
    )

    return train_scaled_df, test_scaled_df


def onehot(df):

    nominal_cols = [
        "Gender"
    ]

    nominal_cols = [
        c for c in nominal_cols
        if c in df.columns
    ]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df.get("PlacementStatus"),
    )

    if not nominal_cols:

        return (
            pd.DataFrame(index=train_df.index),
            pd.DataFrame(index=test_df.index),
        )

    try:

        ohe = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )

    except TypeError:

        ohe = OneHotEncoder(
            drop="first",
            sparse=False,
            handle_unknown="ignore",
        )

    train_ohe = ohe.fit_transform(
        train_df[nominal_cols]
    )

    test_ohe = ohe.transform(
        test_df[nominal_cols]
    )

    feature_names = ohe.get_feature_names_out(
        nominal_cols
    )

    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=feature_names,
        index=train_df.index,
    )

    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=feature_names,
        index=test_df.index,
    )

    return train_ohe_df, test_ohe_df


def ordinal(df):

    df_copy = df.copy()

    if "CGPA" not in df_copy.columns:
        return df_copy, df_copy.copy()

    df_copy["CGPA_Category"] = pd.cut(
        df_copy["CGPA"],
        bins=[0, 6, 8, 10],
        labels=[
            "low",
            "medium",
            "high",
        ],
        ordered=True,
    )

    ordinal_cols = [
        "CGPA_Category"
    ]

    train_df, test_df = train_test_split(
        df_copy,
        test_size=0.2,
        random_state=42,
        stratify=df_copy.get("PlacementStatus"),
    )

    encoder = OrdinalEncoder(
        categories=[
            [
                "low",
                "medium",
                "high",
            ]
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    train_df_encoded[
        ordinal_cols
    ] = encoder.fit_transform(
        train_df[ordinal_cols]
    )

    test_df_encoded[
        ordinal_cols
    ] = encoder.transform(
        test_df[ordinal_cols]
    )

    return (
        train_df_encoded,
        test_df_encoded,
    )


def label_encode_branch(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df.get("PlacementStatus"),
    )

    if "Stream" not in df.columns:

        return (
            train_df,
            test_df,
            {},
        )

    encoder = LabelEncoder()

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    train_df_encoded["Stream"] = (
        encoder.fit_transform(
            train_df["Stream"]
        )
    )

    test_df_encoded["Stream"] = (
        encoder.transform(
            test_df["Stream"]
        )
    )

    mapping = dict(
        zip(
            encoder.classes_,
            encoder.transform(
                encoder.classes_
            ),
        )
    )

    return (
        train_df_encoded,
        test_df_encoded,
        mapping,
    )


def create_cgpa_category(df):

    df_copy = df.copy()

    if "CGPA" in df_copy.columns:

        df_copy["CGPA_Category"] = pd.cut(
            df_copy["CGPA"],
            bins=[0, 6, 8, 10],
            labels=[
                "low",
                "medium",
                "high",
            ],
            ordered=True,
        )

    return df_copy


def find_null_values(df, column):

    if column not in df.columns:

        print(
            f"Column '{column}' not found."
        )

        return

    null_values = df[column].isnull()

    print(
        "Number of NULL values:",
        null_values.sum(),
    )

    print(
        "\nRows where value is NULL:"
    )

    print(
        df[null_values]
    )


def data_cleaning(df):

    cleaned = df.copy()

    if "Workshops" in cleaned.columns:

        cleaned["Workshops"] = (
            cleaned["Workshops"].fillna(
                cleaned["Workshops"].mean()
            )
        )

    if "AptitudeTestScore" in cleaned.columns:

        cleaned["AptitudeTestScore"] = (
            cleaned[
                "AptitudeTestScore"
            ].fillna(
                cleaned[
                    "AptitudeTestScore"
                ].median()
            )
        )

    numerical_cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
    ]

    numerical_cols = [
        c for c in numerical_cols
        if c in cleaned.columns
    ]

    if numerical_cols:

        imputer = KNNImputer(
            n_neighbors=3
        )

        cleaned[numerical_cols] = (
            imputer.fit_transform(
                cleaned[numerical_cols]
            )
        )

    return cleaned


def get_numerical_features(df):

    num_cols = df.select_dtypes(
        include="number"
    ).columns

    return df[num_cols]


def get_categorical_features(df):

    cat_cols = df.select_dtypes(
        include="object"
    ).columns

    return df[cat_cols]


def detect_outliers(df, column):

    if column not in df.columns:
        return pd.DataFrame()

    q1 = df[column].quantile(
        0.25
    )

    q3 = df[column].quantile(
        0.75
    )

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    return df[
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    ]


def select_model_features(df):

    features = [
        "CGPA",
        "Internships",
        "Projects",
        "CodingTestScore",
    ]

    features = [
        f for f in features
        if f in df.columns
    ]

    X = df[features]
    y = df["PlacementStatus"]

    return X, y


# ============================================================
# KNN IMPUTATION
# ============================================================

def knn_impute(df, n_neighbors=3):

    filled = df.copy()

    if "Workshops" in filled.columns:

        filled["Workshops"] = (
            filled[
                "Workshops"
            ].fillna(
                filled[
                    "Workshops"
                ].mean()
            )
        )

    if "AptitudeTestScore" in filled.columns:

        filled[
            "AptitudeTestScore"
        ] = (
            filled[
                "AptitudeTestScore"
            ].fillna(
                filled[
                    "AptitudeTestScore"
                ].median()
            )
        )

    numerical_cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
    ]

    numerical_cols = [
        c for c in numerical_cols
        if c in filled.columns
    ]

    if numerical_cols:

        imputer = KNNImputer(
            n_neighbors=n_neighbors
        )

        filled[numerical_cols] = (
            imputer.fit_transform(
                filled[numerical_cols]
            )
        )

    return {
        "before": int(
            df.isnull().sum().sum()
        ),
        "after": int(
            filled.isnull().sum().sum()
        ),
        "n_neighbors": n_neighbors,
        "columns": numerical_cols,
        "df": filled,
    }


# ============================================================
# LINEAR REGRESSION
# ============================================================

def simple_linear_regression(df):

    X = df[["CGPA"]]
    y = df["Salary Package"]

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    linear_regression = LinearRegression()

    linear_regression.fit(
        train_X,
        train_y,
    )

    y_pred = linear_regression.predict(
        test_X
    )

    mse1 = mean_squared_error(
        test_y,
        y_pred,
    )

    rsq1 = r2_score(
        test_y,
        y_pred,
    )

    print(
        "Intercept (theta0):",
        linear_regression.intercept_,
    )

    print(
        "Slope (theta1):",
        linear_regression.coef_[0],
    )

    print(
        "MSE:",
        mse1,
    )

    print(
        "RSQ:",
        rsq1,
    )


# ============================================================
# GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.001,
    iterations=100,
):

    X = np.asarray(
        X,
        dtype=float,
    ).ravel()

    y = np.asarray(
        y,
        dtype=float,
    ).ravel()

    m = len(y)

    theta0 = 0.0
    theta1 = 0.0

    mse_history = []

    for _ in range(iterations):

        y_pred = (
            theta0
            + theta1 * X
        )

        error = y_pred - y

        mse = np.mean(
            error ** 2
        )

        mse_history.append(
            mse
        )

        d_theta0 = (
            2 / m
        ) * np.sum(error)

        d_theta1 = (
            2 / m
        ) * np.sum(
            error * X
        )

        theta0 -= (
            learning_rate
            * d_theta0
        )

        theta1 -= (
            learning_rate
            * d_theta1
        )

    return (
        theta0,
        theta1,
        mse_history,
    )


def simple_linear_regression_GD(df):

    X = df[["CGPA"]]

    if df["PlacementStatus"].dtype == "object":

        y = (
            df["PlacementStatus"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "placed": 1,
                "unplaced": 0,
            })
            .fillna(0)
        )

    else:

        y = df[
            "PlacementStatus"
        ]

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    theta0, theta1, _ = (
        gradient_descent(
            train_X,
            train_y,
            learning_rate=0.001,
            iterations=100,
        )
    )

    y_pred = (
        theta0
        + theta1
        * np.asarray(
            test_X,
            dtype=float,
        ).ravel()
    )

    mse1 = mean_squared_error(
        test_y,
        y_pred,
    )

    rsq1 = r2_score(
        test_y,
        y_pred,
    )

    return (
        theta0,
        theta1,
        mse1,
        rsq1,
    )


# ============================================================
# MULTIPLE REGRESSION MODELS
# ============================================================

def run_regression_models(df):

    features = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "SoftSkillsRating",
        "AttendancePercent",
    ]

    target = "Salary Package"

    features = [
        c for c in features
        if c in df.columns
    ]

    if target not in df.columns:

        return {
            "target": target,
            "features": features,
            "samples": 0,
            "results": [],
            "coefficients": [],
            "intercept": 0,
        }

    data = df[
        features + [target]
    ].dropna()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge":
            Ridge(alpha=1.0),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            ),
    }

    results = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        preds = model.predict(
            X_test
        )

        results.append({

            "model": name,

            "r2": round(
                float(
                    r2_score(
                        y_test,
                        preds,
                    )
                ),
                4,
            ),

            "rmse": round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            preds,
                        )
                    )
                ),
                4,
            ),

            "mae": round(
                float(
                    mean_absolute_error(
                        y_test,
                        preds,
                    )
                ),
                4,
            ),
        })

    linear = LinearRegression()

    linear.fit(
        X_train,
        y_train,
    )

    coefficients = [

        {
            "feature": col,

            "coefficient": round(
                float(coef),
                4,
            ),
        }

        for col, coef in zip(
            features,
            linear.coef_,
        )
    ]

    return {

        "target": target,

        "features": features,

        "samples": int(
            len(data)
        ),

        "results": results,

        "coefficients": coefficients,

        "intercept": round(
            float(
                linear.intercept_
            ),
            4,
        ),
    }


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def simple_logistic_regression_GD(df):

    X = df[["CGPA"]]

    if df["PlacementStatus"].dtype == "object":

        y = (
            df["PlacementStatus"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "placed": 1,
                "unplaced": 0,
            })
        )

    else:

        y = df[
            "PlacementStatus"
        ]

    valid = y.notna()

    X = X[valid]
    y = y[valid]

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        train_X,
        train_y,
    )

    theta0 = model.intercept_[0]
    theta1 = model.coef_[0]

    predictions = model.predict(
        test_X
    )

    accuracy = accuracy_score(
        test_y,
        predictions,
    )

    return {

        "target":
            "PlacementStatus",

        "feature":
            "CGPA",

        "theta0":
            round(
                float(theta0),
                4,
            ),

        "theta1":
            round(
                float(theta1[0]),
                4,
            ),

        "accuracy":
            round(
                float(accuracy),
                4,
            ),

        "train_samples":
            int(
                len(train_X)
            ),

        "test_samples":
            int(
                len(test_X)
            ),
    }


# ============================================================
# FLASK REGRESSION HELPERS
# ============================================================

def run_simple_linear(df):

    buffer = io.StringIO()

    with redirect_stdout(buffer):

        simple_linear_regression(df)

    output = buffer.getvalue()

    mse = None
    rsq = None

    for line in output.splitlines():

        line = line.strip()

        if line.startswith("MSE:"):

            mse = float(
                line.split(
                    ":",
                    1,
                )[1]
            )

        elif line.startswith("RSQ:"):

            rsq = float(
                line.split(
                    ":",
                    1,
                )[1]
            )

    return {

        "target":
            "Salary Package",

        "feature":
            "CGPA",

        "mse":
            mse,

        "rsq":
            rsq,

        "samples":
            int(len(df)),
    }


def run_linear_gd(df):

    theta0, theta1, mse, rsq = (
        simple_linear_regression_GD(df)
    )

    return {

        "target":
            "PlacementStatus",

        "feature":
            "CGPA",

        "theta0":
            round(
                float(theta0),
                4,
            ),

        "theta1":
            round(
                float(theta1),
                4,
            ),

        "mse":
            round(
                float(mse),
                4,
            ),

        "rsq":
            round(
                float(rsq),
                4,
            ),

        "samples":
            int(len(df)),
    }


# ============================================================
# BUILT-IN LINEAR REGRESSION EXAMPLE
# ============================================================

def builtin_linear_sample():

    data = {

        "StudyHours": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ],

        "ExamScore": [
            35,
            45,
            50,
            60,
            65,
            75,
            80,
            90,
            92,
            98,
        ],
    }

    df = pd.DataFrame(data)

    X = df[
        ["StudyHours"]
    ]

    y = df[
        "ExamScore"
    ]

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    model = LinearRegression()

    model.fit(
        train_X,
        train_y,
    )

    y_pred = model.predict(
        test_X
    )

    mse = mean_squared_error(
        test_y,
        y_pred,
    )

    rsq = r2_score(
        test_y,
        y_pred,
    )

    return {

        "x":
            "StudyHours",

        "y":
            "ExamScore",

        "theta0":
            round(
                float(
                    model.intercept_
                ),
                4,
            ),

        "theta1":
            round(
                float(
                    model.coef_[0]
                ),
                4,
            ),

        "mse":
            round(
                float(mse),
                4,
            ),

        "rsq":
            round(
                float(rsq),
                4,
            ),
    }


# ============================================================
# BUILT-IN LOGISTIC REGRESSION EXAMPLE
# ============================================================

def builtin_logistic_sample():

    data = {

        "StudyHours": [
            1.0,
            1.5,
            2.0,
            3.5,
            4.0,
            5.5,
            6.0,
            7.5,
            8.0,
            9.0,
        ],

        "PassStatus": [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ],
    }

    df = pd.DataFrame(data)

    X = df[
        ["StudyHours"]
    ]

    y = df[
        "PassStatus"
    ]

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )
    )

    model = LogisticRegression()

    model.fit(
        train_X,
        train_y,
    )

    y_pred = model.predict(
        test_X
    )

    accuracy = accuracy_score(
        test_y,
        y_pred,
    )

    return {

        "x":
            "StudyHours",

        "y":
            "PassStatus",

        "theta0":
            round(
                float(
                    model.intercept_[0]
                ),
                4,
            ),

        "theta1":
            round(
                float(
                    model.coef_[0][0]
                ),
                4,
            ),

        "accuracy":
            round(
                float(accuracy),
                4,
            ),

        "predictions":
            y_pred.tolist(),

        "actual":
            test_y.values.tolist(),
    }


# ============================================================
# STEP 6
# DECISION TREE + ROC-AUC + CLASSIFICATION METRICS
# ============================================================

def run_decision_tree_roc_auc(df):

    # --------------------------------------------------------
    # 1. Select features
    # --------------------------------------------------------

    features = [
        "CGPA",
        "Internships",
        "Projects",
        "CodingTestScore",
    ]

    existing_features = [
        f for f in features
        if f in df.columns
    ]

    if not existing_features:

        raise ValueError(
            "None of the required classification "
            "features were found in the dataset."
        )

    X = df[
        existing_features
    ].copy()

    # --------------------------------------------------------
    # 2. Check target
    # --------------------------------------------------------

    if "PlacementStatus" not in df.columns:

        raise ValueError(
            "PlacementStatus column not found in dataset."
        )

    target = df[
        "PlacementStatus"
    ]

    # --------------------------------------------------------
    # 3. Convert target to 0 / 1
    # --------------------------------------------------------

    if (
        target.dtype == "object"
        or
        str(target.dtype).startswith("string")
    ):

        y = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
            .map({

                "yes": 1,
                "no": 0,

                "placed": 1,
                "unplaced": 0,

                "not placed": 0,

                "1": 1,
                "0": 0,
            })
        )

    else:

        y = pd.to_numeric(
            target,
            errors="coerce",
        )

    # --------------------------------------------------------
    # 4. Combine and remove invalid rows
    # --------------------------------------------------------

    data = X.copy()

    data["target"] = y

    data = data.dropna()

    if len(data) < 10:

        raise ValueError(
            "Not enough valid rows for classification."
        )

    X = data[
        existing_features
    ]

    y = data[
        "target"
    ].astype(int)

    # --------------------------------------------------------
    # 5. Make sure target is binary
    # --------------------------------------------------------

    unique_classes = sorted(
        y.unique().tolist()
    )

    if len(unique_classes) != 2:

        raise ValueError(
            "PlacementStatus must contain exactly "
            "two classes."
        )

    # --------------------------------------------------------
    # 6. Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # --------------------------------------------------------
    # 7. Decision Tree
    # --------------------------------------------------------

    dt_model = DecisionTreeClassifier(

        max_depth=8,

        min_samples_split=10,

        min_samples_leaf=5,

        random_state=42,
    )

    dt_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 8. Predictions
    # --------------------------------------------------------

    y_pred = dt_model.predict(
        X_test
    )

    # Probability of positive class
    y_prob = dt_model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # 9. Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # 10. Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 11. Recall / Sensitivity
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 12. F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 13. Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    # --------------------------------------------------------
    # 14. Specificity
    #
    # Specificity = TN / (TN + FP)
    # --------------------------------------------------------

    if (tn + fp) > 0:

        specificity = (
            tn / (tn + fp)
        )

    else:

        specificity = 0.0

    # --------------------------------------------------------
    # 15. ROC Curve
    # --------------------------------------------------------

    fpr, tpr, thresholds = (
        roc_curve(
            y_test,
            y_prob,
        )
    )

    # --------------------------------------------------------
    # 16. ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

    # --------------------------------------------------------
    # 17. Create ROC graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"Decision Tree "
            f"(AUC = {roc_auc:.3f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Guess",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate (Recall)"
    )

    plt.title(
        "ROC Curve - Placement Prediction"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # 18. Convert graph to Base64
    # --------------------------------------------------------

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    buffer.seek(0)

    plot_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    plot_url = (
        "data:image/png;base64,"
        + plot_base64
    )

    # --------------------------------------------------------
    # 19. Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 20. Interpretation
    # --------------------------------------------------------

    if roc_auc >= 0.90:

        discrimination = "Excellent"

    elif roc_auc >= 0.80:

        discrimination = "Good"

    elif roc_auc >= 0.70:

        discrimination = "Fair"

    elif roc_auc >= 0.60:

        discrimination = "Poor"

    else:

        discrimination = "Near Random"

    # --------------------------------------------------------
    # 21. Return everything to Flask
    # --------------------------------------------------------

    return {

        # Classification metrics
        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "specificity":
            float(specificity),

        # ROC
        "roc_auc":
            float(roc_auc),

        "roc_plot":
            plot_url,

        "plot_url":
            plot_url,

        # Confusion matrix
        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        # Classification report
        "report":
            report,

        # Model information
        "features":
            existing_features,

        "split_ratio":
            {
                "train": 0.80,
                "test": 0.20,
            },

        "train_samples":
            int(len(X_train)),

        "test_samples":
            int(len(X_test)),

        "total_samples":
            int(len(data)),

        # Interpretation
        "discrimination":
            discrimination,
    }