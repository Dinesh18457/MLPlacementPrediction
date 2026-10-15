import matplotlib

matplotlib.use("Agg")

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from pathlib import Path


PLOTS_DIR = Path("static/plots")


def _plot_exists(*names):

    return all(
        (
            PLOTS_DIR /
            name
        ).exists()
        for name in names
    )


PLOT_FILENAMES = [
    "placement_distribution.png",
    "cgpa_placement.png",
    "correlation_heatmap.png",
    "college_placement.png",
    "internships_placement.png",
    "missing_heatmap.png",
    "numeric_histograms.png",
    "outlier_boxplots.png",
    "relationship_scatter.png",
    "categorical_counts.png",
    "gender_placement.png",
    "stream_placement.png",
    "sgpa_trend.png",
    "salary_distribution.png",
    "salary_by_college.png",
    "pairplot.png"
]


def summarize(df):

    return {

        "shape":
            df.shape,

        "columns":
            df.columns.tolist(),

        "numeric_columns":
            df.select_dtypes(
                include="number"
            ).columns.tolist(),

        "categorical_columns":
            df.select_dtypes(
                include="object"
            ).columns.tolist(),

        "duplicates":
            int(
                df.duplicated().sum()
            )

    }


def column_summary(df, column):

    series = df[column].dropna()

    return {

        "mean":
            series.mean(),

        "median":
            series.median(),

        "mode":
            series.mode().tolist(),

        "minimum":
            series.min(),

        "maximum":
            series.max(),

        "range":
            series.max()
            -
            series.min(),

        "variance":
            series.var(),

        "standard_deviation":
            series.std(),

        "q1":
            series.quantile(0.25),

        "q3":
            series.quantile(0.75),

        "iqr":
            series.quantile(0.75)
            -
            series.quantile(0.25)

    }


def get_correlation_with_target(df):

    numeric_df = df.select_dtypes(
        include="number"
    ).copy()

    if "PlacementStatus" not in numeric_df.columns:

        if "PlacementStatus" in df.columns:

            numeric_df["PlacementStatus"] = (
                df["PlacementStatus"]
                .astype("category")
                .cat.codes
            )

    correlation = (
        numeric_df
        .corr()["PlacementStatus"]
        .drop("PlacementStatus")
        .sort_values(
            ascending=False
        )
    )

    return [

        {
            "feature":
                index,

            "correlation":
                round(
                    float(value),
                    3
                )

        }

        for index, value
        in correlation.items()

    ]


def cgpa_placement_ttest(df):

    if "PlacementStatus" not in df.columns:

        return stats.ttest_ind(
            [],
            []
        )


    placed = df[
        df["PlacementStatus"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "yes",
                "1",
                "placed"
            ]
        )
    ]["CGPA"].dropna()


    not_placed = df[
        ~df["PlacementStatus"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "yes",
                "1",
                "placed"
            ]
        )
    ]["CGPA"].dropna()


    return stats.ttest_ind(
        placed,
        not_placed,
        equal_var=False
    )


def generate_all_plots(df):

    if _plot_exists(
        "placement_distribution.png",
        "cgpa_placement.png",
        "correlation_heatmap.png",
        "college_placement.png",
        "internships_placement.png"
    ):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="PlacementStatus"
    )

    plt.title(
        "Placement Status Distribution"
    )

    plt.xlabel(
        "Placement Status"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "placement_distribution.png"
    )

    plt.close()


    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="CGPA",
        hue="PlacementStatus",
        kde=True
    )

    plt.title(
        "CGPA Distribution by Placement Status"
    )

    plt.xlabel(
        "CGPA"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "cgpa_placement.png"
    )

    plt.close()


    numeric_df = df.select_dtypes(
        include="number"
    )

    plt.figure(
        figsize=(12, 8)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "correlation_heatmap.png"
    )

    plt.close()


    if "CollegeTier" in df.columns:

        placement_rate = (
            df.groupby(
                "CollegeTier"
            )["PlacementStatus"]
            .apply(
                lambda x:
                x.astype(str)
                .str.lower()
                .isin(
                    [
                        "yes",
                        "1",
                        "placed"
                    ]
                )
                .mean()
            )
        )


        plt.figure(
            figsize=(8, 5)
        )

        placement_rate.plot(
            kind="bar"
        )

        plt.title(
            "Placement Rate by College Tier"
        )

        plt.xlabel(
            "College Tier"
        )

        plt.ylabel(
            "Placement Rate"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "college_placement.png"
        )

        plt.close()


    if "Internships" in df.columns:

        placement_rate = (
            df.groupby(
                "Internships"
            )["PlacementStatus"]
            .apply(
                lambda x:
                x.astype(str)
                .str.lower()
                .isin(
                    [
                        "yes",
                        "1",
                        "placed"
                    ]
                )
                .mean()
            )
        )


        plt.figure(
            figsize=(8, 5)
        )

        placement_rate.plot(
            kind="bar"
        )

        plt.title(
            "Placement Rate by Number of Internships"
        )

        plt.xlabel(
            "Internships"
        )

        plt.ylabel(
            "Placement Rate"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "internships_placement.png"
        )

        plt.close()


def get_overview_stats(df):

    placement_values = (
        df["PlacementStatus"]
        .astype(str)
        .str.lower()
    )


    placed_count = placement_values.isin(
        [
            "yes",
            "1",
            "placed"
        ]
    ).sum()


    placement_rate = round(
        (
            placed_count /
            len(df)
        ) * 100,
        2
    )


    return {

        "placement_rate":
            placement_rate,

        "rows":
            df.shape[0],

        "columns":
            df.shape[1],

        "duplicates":
            int(
                df.duplicated().sum()
            ),

        "missing":
            int(
                df.isnull()
                .sum()
                .sum()
            )

    }


def get_univariate_stats(df):

    result = []


    for column in df.select_dtypes(
        include="number"
    ).columns:

        summary = column_summary(
            df,
            column
        )


        result.append({

            "feature":
                column,

            "mean":
                round(
                    summary["mean"],
                    2
                ),

            "median":
                round(
                    summary["median"],
                    2
                ),

            "mode":
                summary["mode"],

            "minimum":
                round(
                    summary["minimum"],
                    2
                ),

            "maximum":
                round(
                    summary["maximum"],
                    2
                ),

            "range":
                round(
                    summary["range"],
                    2
                ),

            "variance":
                round(
                    summary["variance"],
                    2
                ),

            "std":
                round(
                    summary[
                        "standard_deviation"
                    ],
                    2
                ),

            "q1":
                round(
                    summary["q1"],
                    2
                ),

            "q3":
                round(
                    summary["q3"],
                    2
                ),

            "iqr":
                round(
                    summary["iqr"],
                    2
                )

        })


    return result


def get_bivariate_stats(df):

    return get_correlation_with_target(
        df
    )


def get_multivariate_stats(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    return {

        "columns":
            numeric_df.columns.tolist(),

        "correlation":
            numeric_df.corr()
            .round(3)
            .to_dict()

    }


def get_correlation_stats(df):

    return get_correlation_with_target(
        df
    )


# =========================================================
# EDA METHODS (Tasks 1-15)
# =========================================================

NUMERIC_FEATURES = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",
    "SoftSkillsRating",
    "MockInterviewScore"
]

SCORE_FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier"
]

SGPA_COLUMNS = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8"
]

PLOT_FILENAMES = [
    "placement_distribution.png",
    "cgpa_placement.png",
    "correlation_heatmap.png",
    "college_placement.png",
    "internships_placement.png",
    "missing_heatmap.png",
    "numeric_histograms.png",
    "outlier_boxplots.png",
    "relationship_scatter.png",
    "categorical_counts.png",
    "gender_placement.png",
    "stream_placement.png",
    "sgpa_trend.png",
    "salary_distribution.png",
    "salary_by_college.png",
    "pairplot.png"
]


def _is_placed_series(df):

    return (
        df["PlacementStatus"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "yes",
                "1",
                "placed"
            ]
        )
    )


def _placement_labels(df):

    return _is_placed_series(df).map(
        {
            True: "Placed",
            False: "Not Placed"
        }
    )


# ---- Task 2: Basic structure / describe ----

def get_numeric_describe(df):

    return df.select_dtypes(
        include="number"
    ).describe().round(2).to_dict()


def get_categorical_summary(df):

    result = []

    for column in df.select_dtypes(
        include="object"
    ).columns:

        series = df[column]

        if series.notna().sum() == 0:
            continue

        top = series.mode()

        result.append({

            "column": column,

            "count":
                int(series.count()),

            "unique":
                int(series.nunique()),

            "top":
                top[0]
                if len(top) > 0
                else "N/A",

            "freq":
                int(
                    series.value_counts().iloc[0]
                )

        })

    return result


# ---- Task 3: Missing values ----

def get_missing_summary(df):

    result = []

    for column in df.columns:

        count = int(
            df[column].isnull().sum()
        )

        percentage = round(
            (
                count
                /
                len(df)
            ) * 100,
            2
        )

        result.append({

            "column": column,

            "type":
                str(df[column].dtype),

            "count":
                count,

            "percentage":
                percentage

        })

    return result


def median_fill_numeric(df):

    filled = df.copy()

    numeric_cols = filled.select_dtypes(
        include="number"
    ).columns

    for column in numeric_cols:

        filled[column] = filled[column].fillna(
            filled[column].median()
        )

    return filled


def plot_missing_heatmap(df):

    if _plot_exists("missing_heatmap.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_size = min(
        len(df),
        5000
    )

    missing = (
        df.isnull()
        .sample(
            n=sample_size,
            random_state=42
        )
    )

    plt.figure(
        figsize=(10, 4)
    )

    sns.heatmap(
        missing,
        cbar=False,
        yticklabels=False,
        cmap="viridis"
    )

    plt.title(
        "Missing Values Heatmap"
    )

    plt.xlabel(
        "Columns"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "missing_heatmap.png",
        dpi=100
    )

    plt.close()


# ---- Task 4: Duplicate rows ----

def get_duplicate_stats(df):

    count = int(
        df.duplicated().sum()
    )

    return {

        "count":
            count,

        "percentage":
            round(
                (
                    count
                    /
                    len(df)
                ) * 100,
                2
            )

    }


# ---- Task 6: Numeric feature distributions ----

def plot_numeric_histograms(df):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    available = [
        col
        for col in NUMERIC_FEATURES
        if col in df.columns
    ]

    filenames = [
        f"hist_{col}.png"
        for col in available
    ]

    if _plot_exists(*filenames):

        return

    for col, filename in zip(
        available,
        filenames
    ):

        plt.figure(
            figsize=(4, 3)
        )

        sns.histplot(
            df[col].dropna(),
            kde=True
        )

        plt.title(col)

        plt.xlabel("")

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            filename,
            dpi=100
        )

        plt.close()


# ---- Task 7: Outlier detection (boxplots) ----

def plot_outlier_boxplots(df):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        col
        for col in (
            SCORE_FEATURES +
            ["Salary Package"]
        )
        if col in df.columns
    ]

    filenames = [
        f"box_{col.replace(' ', '_')}.png"
        for col in columns
    ]

    if _plot_exists(*filenames):

        return

    for col, filename in zip(
        columns,
        filenames
    ):

        plt.figure(
            figsize=(4, 3)
        )

        sns.boxplot(
            x=df[col].dropna()
        )

        plt.title(col)

        plt.xlabel("")

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            filename,
            dpi=100
        )

        plt.close()


# ---- Task 8: Correlation analysis (heatmap + target) ----

def plot_correlation_heatmap(df):

    if _plot_exists("correlation_heatmap.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "correlation_heatmap.png",
        dpi=100
    )

    plt.close()


# ---- Task 9: Relationship scatter plots ----

def plot_relationship_scatter(df):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        "CGPA" in df.columns
        and
        "Salary Package" in df.columns
    ):

        plt.figure(
            figsize=(5, 4)
        )

        sns.scatterplot(
            data=df,
            x="CGPA",
            y="Salary Package",
            alpha=0.3
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "scatter_cgpa_salary.png",
            dpi=100
        )

        plt.close()

    if (
        "AptitudeTestScore" in df.columns
        and
        "CodingTestScore" in df.columns
    ):

        plt.figure(
            figsize=(5, 4)
        )

        sns.scatterplot(
            data=df,
            x="AptitudeTestScore",
            y="CodingTestScore",
            alpha=0.3
        )

        plt.title(
            "Aptitude vs Coding Test Score"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "scatter_aptitude_coding.png",
            dpi=100
        )

        plt.close()


# ---- Task 10: Categorical feature counts ----

def plot_categorical_counts(df):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    available = [
        col
        for col in CATEGORICAL_FEATURES
        if col in df.columns
    ]

    filenames = [
        f"cat_{col}.png"
        for col in available
    ]

    if _plot_exists(*filenames):

        return

    for col, filename in zip(
        available,
        filenames
    ):

        counts = (
            df[col]
            .astype(str)
            .value_counts()
        )

        if col == "City":
            counts = counts.head(15)

        plt.figure(
            figsize=(5, 3.2)
        )

        counts.plot(
            kind="bar"
        )

        plt.title(col)

        plt.xlabel("")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            filename,
            dpi=100
        )

        plt.close()


# ---- Task 11: Gender vs Placement Status ----

def plot_gender_placement(df):

    if _plot_exists("gender_placement.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(5, 4)
    )

    sns.countplot(
        data=df,
        x="Gender",
        hue=_placement_labels(df)
    )

    plt.title(
        "Placement Status by Gender"
    )

    plt.xlabel(
        "Gender"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "gender_placement.png",
        dpi=100
    )

    plt.close()


# ---- Task 12: College tier / Stream vs Placement Status ----

def plot_stream_placement(df):

    if _plot_exists("stream_placement.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    placement_rate = (
        df.groupby("Stream")
        ["PlacementStatus"]
        .apply(
            lambda x:
            x.astype(str)
            .str.lower()
            .isin(
                [
                    "yes",
                    "1",
                    "placed"
                ]
            )
            .mean()
        )
    )

    plt.figure(
        figsize=(6, 4)
    )

    placement_rate.sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Placement Rate by Stream"
    )

    plt.xlabel(
        "Stream"
    )

    plt.ylabel(
        "Placement Rate"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "stream_placement.png",
        dpi=100
    )

    plt.close()


def get_placement_breakdown(df, column):

    if column not in df.columns:

        return []

    table = (
        df.groupby(
            [
                column,
                _placement_labels(df)
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    return table.to_dict(
        orient="records"
    )


# ---- Task 13: SGPA trend across semesters ----

def plot_sgpa_trend(df):

    if _plot_exists("sgpa_trend.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    available = [
        col
        for col in SGPA_COLUMNS
        if col in df.columns
    ]

    trend = (
        df.groupby(
            _placement_labels(df)
        )[available]
        .mean()
    )

    plt.figure(
        figsize=(7, 4)
    )

    for label in trend.index:

        plt.plot(
            available,
            trend.loc[label],
            marker="o",
            label=label
        )

    plt.title(
        "Average SGPA Trend Across Semesters"
    )

    plt.xlabel(
        "Semester"
    )

    plt.ylabel(
        "Average SGPA"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "sgpa_trend.png",
        dpi=100
    )

    plt.close()


def get_sgpa_trend_data(df):

    available = [
        col
        for col in SGPA_COLUMNS
        if col in df.columns
    ]

    trend = (
        df.groupby(
            _placement_labels(df)
        )[available]
        .mean()
        .round(3)
    )

    return {

        "semesters":
            available,

        "placed":
            trend.loc["Placed"].tolist()
            if "Placed" in trend.index
            else [],

        "not_placed":
            trend.loc["Not Placed"].tolist()
            if "Not Placed" in trend.index
            else []

    }


# ---- Task 14: Salary package analysis ----

def plot_salary_analysis(df):

    if _plot_exists(
        "salary_distribution.png",
        "salary_by_college.png"
    ):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    placed = df[_is_placed_series(df)]

    if "Salary Package" not in placed.columns:

        return

    plt.figure(
        figsize=(5, 4)
    )

    sns.histplot(
        placed["Salary Package"].dropna(),
        kde=True
    )

    plt.title(
        "Salary Package Distribution (Placed Students)"
    )

    plt.xlabel(
        "Salary Package (LPA)"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "salary_distribution.png",
        dpi=100
    )

    plt.close()

    if "CollegeTier" in placed.columns:

        plt.figure(
            figsize=(5, 4)
        )

        sns.boxplot(
            data=placed,
            x="CollegeTier",
            y="Salary Package"
        )

        plt.title(
            "Salary Package by College Tier (Placed Students)"
        )

        plt.xlabel(
            "College Tier"
        )

        plt.ylabel(
            "Salary Package (LPA)"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "salary_by_college.png",
            dpi=100
        )

        plt.close()


def get_salary_stats(df):

    placed = df[_is_placed_series(df)]

    if "Salary Package" not in placed.columns:

        return {}

    series = placed["Salary Package"].dropna()

    if series.empty:

        return {}

    return {

        "count":
            int(series.count()),

        "mean":
            round(series.mean(), 2),

        "median":
            round(series.median(), 2),

        "std":
            round(series.std(), 2),

        "minimum":
            round(series.min(), 2),

        "maximum":
            round(series.max(), 2)

    }


# ---- Task 15: Pairplot ----

def plot_pairplot(df):

    if _plot_exists("pairplot.png"):

        return

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sample = df.sample(
        n=min(
            len(df),
            2000
        ),
        random_state=42
    )

    available = [
        col
        for col in SCORE_FEATURES
        if col in sample.columns
    ]

    data = sample[available].copy()

    data["Placement"] = _placement_labels(sample)

    g = sns.pairplot(
        data,
        hue="Placement",
        corner=True,
        height=1.8
    )

    g.fig.suptitle(
        "Pairplot of Key Scores by Placement Status",
        y=1.02
    )

    g.fig.savefig(
        PLOTS_DIR /
        "pairplot.png",
        bbox_inches="tight",
        dpi=100
    )

    plt.close(g.fig)


# ---- Run every plot for the EDA dashboard ----

def generate_full_eda_plots(df):

    generate_all_plots(df)

    plot_correlation_heatmap(df)

    plot_missing_heatmap(df)

    plot_numeric_histograms(df)

    plot_outlier_boxplots(df)

    plot_relationship_scatter(df)

    plot_categorical_counts(df)

    plot_gender_placement(df)

    plot_stream_placement(df)

    plot_sgpa_trend(df)

    plot_salary_analysis(df)

    plot_pairplot(df)