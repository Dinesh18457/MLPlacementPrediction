from flask import Flask, render_template
from sklearn.tree import DecisionTreeClassifier
from models.load import load_data

from models.feature_engg import (
    knn_impute,
    standardization,
    simple_logistic_regression_GD,
    run_regression_models,
    run_simple_linear,
    run_linear_gd,
    builtin_linear_sample,
    builtin_logistic_sample,
    run_decision_tree_roc_auc
)

from models.eda import (
    summarize,
    get_correlation_with_target,
    column_summary,
    cgpa_placement_ttest,
    get_missing_summary,
    get_numeric_describe,
    get_categorical_summary,
    get_duplicate_stats,
    get_univariate_stats,
    get_placement_breakdown,
    get_sgpa_trend_data,
    get_salary_stats,
    median_fill_numeric,
    generate_full_eda_plots
)


app = Flask(__name__)

_cache = {}


# ============================================================
# CACHE HELPER
# ============================================================

def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


# ============================================================
# LOAD DATA + MODELS
# ============================================================

def _load_and_cache():

    df = load_data()

    knn = _cached(
        "knn",
        lambda: knn_impute(df)
    )

    regression = _cached(
        "regression",
        lambda: run_regression_models(df)
    )

    # Decision Tree + ROC-AUC
    dt_roc = _cached(
        "dt_roc",
        lambda: run_decision_tree_roc_auc(knn["df"])
    )

    return df, knn, regression, dt_roc


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# STEP 1 - DATA LOADING
# ============================================================

@app.route("/load")
def load_page():

    df, knn, _, _ = _load_and_cache()

    cleaned_df = knn["df"]

    summary = summarize(df)

    cleaned_summary = summarize(
        cleaned_df
    )

    return render_template(
        "load.html",

        shape=summary["shape"],

        columns=summary["columns"],

        preview=df.head(10).to_dict(
            orient="records"
        ),

        cleaned_shape=cleaned_summary["shape"],

        cleaned_columns=cleaned_summary["columns"],

        cleaned_preview=cleaned_df.head(10).to_dict(
            orient="records"
        )
    )


# ============================================================
# STEP 2 - EDA
# ============================================================

@app.route("/eda")
def eda_page():

    df, knn, regression, _ = _load_and_cache()

    summary = summarize(df)

    generate_full_eda_plots(df)

    # --------------------------------------------------------
    # Placement percentage
    # --------------------------------------------------------

    placement_values = (
        df["PlacementStatus"]
        .astype(str)
        .str.lower()
    )

    placed_count = placement_values.isin(
        ["yes", "1", "placed"]
    ).sum()

    placement_rate = round(
        (placed_count / len(df)) * 100,
        2
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = get_missing_summary(df)

    duplicate_stats = get_duplicate_stats(df)

    filled_df = median_fill_numeric(df)

    missing_before = int(
        df.isnull().sum().sum()
    )

    missing_after = int(
        filled_df.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    unique_values = []

    for column in df.columns:

        unique_values.append(
            {
                "column": column,
                "count": int(
                    df[column].nunique()
                )
            }
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    descriptive_statistics = get_univariate_stats(df)

    numeric_describe = get_numeric_describe(df)

    categorical_summary = get_categorical_summary(df)

    # --------------------------------------------------------
    # CGPA statistics
    # --------------------------------------------------------

    cgpa_values = column_summary(
        df,
        "CGPA"
    )

    cgpa_mode = cgpa_values["mode"]

    if len(cgpa_mode) > 0:
        cgpa_mode = cgpa_mode[0]
    else:
        cgpa_mode = "N/A"

    cgpa = {

        "mean": round(
            cgpa_values["mean"],
            2
        ),

        "median": round(
            cgpa_values["median"],
            2
        ),

        "mode": cgpa_mode,

        "range": round(
            cgpa_values["range"],
            2
        ),

        "variance": round(
            cgpa_values["variance"],
            2
        ),

        "std": round(
            cgpa_values["standard_deviation"],
            2
        ),

        "q1": round(
            cgpa_values["q1"],
            2
        ),

        "q3": round(
            cgpa_values["q3"],
            2
        ),

        "iqr": round(
            cgpa_values["iqr"],
            2
        )
    }

    # --------------------------------------------------------
    # T-test
    # --------------------------------------------------------

    ttest_result = cgpa_placement_ttest(df)

    ttest = {

        "statistic": round(
            float(ttest_result.statistic),
            4
        ),

        "pvalue": round(
            float(ttest_result.pvalue),
            6
        )
    }

    # --------------------------------------------------------
    # Placement breakdowns
    # --------------------------------------------------------

    gender_breakdown = get_placement_breakdown(
        df,
        "Gender"
    )

    college_breakdown = get_placement_breakdown(
        df,
        "CollegeTier"
    )

    stream_breakdown = get_placement_breakdown(
        df,
        "Stream"
    )

    sgpa_trend = get_sgpa_trend_data(df)

    salary_stats = get_salary_stats(df)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    train_scaled, test_scaled = standardization(
        knn["df"]
    )

    std_columns = train_scaled.columns.tolist()

    std_train_rows = (
        train_scaled
        .head(5)
        .round(3)
        .values
        .tolist()
    )

    std_test_rows = (
        test_scaled
        .head(5)
        .round(3)
        .values
        .tolist()
    )

    return render_template(

        "eda.html",

        shape=summary["shape"],

        columns=summary["columns"],

        numeric_columns=summary["numeric_columns"],

        categorical_columns=summary["categorical_columns"],

        duplicates=summary["duplicates"],

        total_missing=missing_before,

        total_unique=int(
            df.nunique().sum()
        ),

        missing_values=missing_values,

        missing_before=missing_before,

        missing_after=missing_after,

        duplicate_stats=duplicate_stats,

        unique_values=unique_values,

        descriptive_statistics=descriptive_statistics,

        numeric_describe=numeric_describe,

        categorical_summary=categorical_summary,

        correlations=get_correlation_with_target(df),

        cgpa=cgpa,

        ttest=ttest,

        placement_rate=placement_rate,

        preview=df.head(10).to_dict(
            orient="records"
        ),

        gender_breakdown=gender_breakdown,

        college_breakdown=college_breakdown,

        stream_breakdown=stream_breakdown,

        sgpa_trend=sgpa_trend,

        salary_stats=salary_stats,

        knn=knn,

        regression=regression,

        std_columns=std_columns,

        std_train_rows=std_train_rows,

        std_test_rows=std_test_rows
    )


# ============================================================
# STEP 3 - REGRESSION
# ============================================================

@app.route("/regression")
def regression_page():

    df, knn, regression, _ = _load_and_cache()

    linear_df = knn["df"].dropna(
        subset=["Salary Package"]
    )

    data = {

        "linear": _cached(
            "slr",
            lambda: run_simple_linear(linear_df)
        ),

        "gd": _cached(
            "slr_gd",
            lambda: run_linear_gd(knn["df"])
        ),

        "logistic": _cached(
            "logi",
            lambda: simple_logistic_regression_GD(
                knn["df"]
            )
        ),

        "builtin_linear": _cached(
            "blin",
            builtin_linear_sample
        ),

        "builtin_logistic": _cached(
            "blog",
            builtin_logistic_sample
        ),

        "multi": regression
    }

    return render_template(

        "regression.html",

        linear=data["linear"],

        gd=data["gd"],

        logistic=data["logistic"],

        builtin_linear=data["builtin_linear"],

        builtin_logistic=data["builtin_logistic"],

        multi=data["multi"]
    )


# ============================================================
# STEP 4, 5, 6 - DECISION TREE + ROC-AUC
# ============================================================

@app.route("/classification")
def classification_page():

    df, knn, _, dt_roc = _load_and_cache()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = float(
        dt_roc.get(
            "accuracy",
            0
        )
    )

    # If accuracy is stored as decimal:
    # 0.896 -> 89.6

    if accuracy <= 1:
        accuracy_percent = round(
            accuracy * 100,
            2
        )
    else:
        accuracy_percent = round(
            accuracy,
            2
        )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = float(
        dt_roc.get(
            "roc_auc",
            0
        )
    )

    roc_auc = round(
        roc_auc,
        3
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = dt_roc.get(
        "report",
        {}
    )

    # --------------------------------------------------------
    # Precision / Recall / F1
    # Use class "1" = Placed
    # --------------------------------------------------------

    class_1 = report.get(
        "1",
        {}
    )

    precision = float(
        class_1.get(
            "precision",
            0
        )
    )

    recall = float(
        class_1.get(
            "recall",
            0
        )
    )

    f1_score = float(
        class_1.get(
            "f1-score",
            0
        )
    )

    # --------------------------------------------------------
    # Specificity
    #
    # If feature_engg.py already calculates it,
    # use that value.
    # --------------------------------------------------------

    specificity = dt_roc.get(
        "specificity",
        None
    )

    if specificity is None:

        # Calculate specificity using class 0 recall.
        #
        # Specificity = TN / (TN + FP)
        #
        # For binary classification, recall of class 0
        # corresponds to specificity for class 1.

        class_0 = report.get(
            "0",
            {}
        )

        specificity = float(
            class_0.get(
                "recall",
                0
            )
        )

    specificity = float(
        specificity
    )

    # --------------------------------------------------------
    # Convert metrics to percentages
    # --------------------------------------------------------

    precision_percent = round(
        precision * 100,
        2
    )

    recall_percent = round(
        recall * 100,
        2
    )

    f1_percent = round(
        f1_score * 100,
        2
    )

    specificity_percent = round(
        specificity * 100,
        2
    )

    # --------------------------------------------------------
    # ROC plot
    # --------------------------------------------------------

    roc_plot = dt_roc.get(
        "plot_url",
        ""
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if roc_auc >= 0.90:
        interpretation = "Excellent"

    elif roc_auc >= 0.80:
        interpretation = "Good"

    elif roc_auc >= 0.70:
        interpretation = "Fair"

    elif roc_auc >= 0.60:
        interpretation = "Poor"

    else:
        interpretation = "Needs Improvement"

    # --------------------------------------------------------
    # Final result object
    # --------------------------------------------------------

    dt_results = {

        "accuracy": accuracy_percent,

        "precision": precision_percent,

        "recall": recall_percent,

        "f1_score": f1_percent,

        "specificity": specificity_percent,

        "roc_auc": roc_auc,

        "roc_plot": roc_plot,

        "report": report,

        "discrimination": dt_roc.get(
            "discrimination",
            interpretation
        ),

        "interpretation": interpretation,

        "features": dt_roc.get(
            "features",
            []
        ),

        "split_ratio": dt_roc.get(
            "split_ratio",
            "70% training / 30% testing"
        )
    }

    return render_template(

        "classification.html",

        dt_results=dt_results
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )