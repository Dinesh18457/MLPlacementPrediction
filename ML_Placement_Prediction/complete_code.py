"""
Placement Prediction ML Pipeline
Author: ML Development Team
Description: Professional implementation of steps 1 through 6 including Decision Tree
             training, probability scoring, ROC-AUC calculation, and curve visualization.
"""

from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, roc_auc_score

# Global Dataset Path Configuration
DATASET_PATH = "D:/ML/placement_predict_50k Dataset.csv"


# =================================================================================
# STEP 1: Load & Prepare Data (Stratified Split & Preprocessing)
# =================================================================================

def load_and_prepare_data(data_path: str):
    """
    Step 1: Loads placement dataset, handles target mapping, applies
    stratified train-test split, and standardizes numeric features.
    """
    df = pd.read_csv(data_path)

    features = ["CGPA", "Internships", "Projects", "CodingTestScore"]
    X = df[features]

    # Map target binary classes (1 for Placed/Yes, 0 for Not Placed/No)
    if df["PlacementStatus"].dtype == "object":
        y = df["PlacementStatus"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    else:
        y = df["PlacementStatus"]

    # Stratified split to maintain class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# =================================================================================
# EXPLORATORY DATA ANALYSIS & PREPROCESSING MODULES
# =================================================================================

def explore_dataset(df: pd.DataFrame):
    """Performs initial exploratory data analysis and summary statistics."""
    print("--- HEAD ---")
    print(df.head())
    print("\n--- INFO ---")
    df.info()
    print("\n--- DESCRIBE ---")
    print(df.describe())
    print("\n--- SHAPE ---")
    print(df.shape)
    print("\n--- NULL VALUES ---")
    print(df.isnull().sum())
    print("\n--- TAIL ---")
    print(df.tail())
    print("\n--- DUPLICATES ---")
    print(df.duplicated().sum())
    print("\n--- UNIQUE VALUES ---")
    print(df.nunique())
    print("\n--- COLUMNS ---")
    print(df.columns.tolist())
    print("=" * 80)


def perform_statistical_analysis(df: pd.DataFrame):
    """Computes summary statistics and hypothesis testing for CGPA."""
    mean_cgpa = df["CGPA"].mean()
    median_cgpa = df["CGPA"].median()
    mode_cgpa = df["CGPA"].mode().tolist()

    print(f"Mean CGPA: {mean_cgpa:.4f}")
    print(f"Median CGPA: {median_cgpa:.4f}")
    print(f"Mode CGPA: {mode_cgpa}")
    print("=" * 80)

    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes("object").columns
    print("Numerical Columns:", num_cols.tolist())
    print("Categorical Columns:", cat_cols.tolist())

    df["CGPA_Category"] = pd.cut(
        df["CGPA"],
        bins=[0, 6, 8, 10],
        labels=["low", "medium", "high"],
        ordered=True
    )
    print("\nCGPA Category Sample:")
    print(df[["CGPA", "CGPA_Category"]].head())
    print("=" * 80)

    # Plot Distribution
    print(df["CGPA"].describe())
    sns.histplot(df["CGPA"], kde=True)
    plt.title("CGPA Distribution")
    plt.show()

    # Hypothesis Testing (Independent T-Test)
    stats_result = stats.ttest_ind(
        df[df["PlacementStatus"] == "Yes"]["CGPA"],
        df[df["PlacementStatus"] == "No"]["CGPA"]
    )
    print("T-Test Result (Placed vs Not Placed CGPA):", stats_result)
    print("=" * 80)

    # Dispersion Metrics
    cgpa_range = df["CGPA"].max() - df["CGPA"].min()
    cgpa_variance = df["CGPA"].var()
    cgpa_std = df["CGPA"].std()
    q1, q3 = df["CGPA"].quantile([0.25, 0.75])
    iqr = q3 - q1

    print(f"CGPA Range: {cgpa_range:.4f}")
    print(f"CGPA Variance: {cgpa_variance:.4f}")
    print(f"CGPA Standard Deviation: {cgpa_std:.4f}")
    print(f"Q1: {q1:.4f} | Q3: {q3:.4f} | IQR: {iqr:.4f}")
    print("=" * 80)


def run_one_hot_encoding(data_path: str):
    """Executes One-Hot Encoding on nominal categorical columns."""
    print("Executing One-Hot Encoding...")
    df = pd.read_csv(data_path)
    nominal_cols = ["Gender"]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"]
    )

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[nominal_cols])
    test_ohe = ohe.transform(test_df[nominal_cols])

    feature_names = ohe.get_feature_names_out(nominal_cols)
    train_ohe_df = pd.DataFrame(train_ohe, columns=feature_names)
    test_ohe_df = pd.DataFrame(test_ohe, columns=feature_names)

    print(train_ohe_df.head())
    print("=" * 80)
    return train_ohe_df, test_ohe_df


def run_ordinal_encoding(data_path: str):
    """Executes Ordinal Encoding on ordered categorical variables."""
    print("Executing Ordinal Encoding...")
    df = pd.read_csv(data_path)

    df["CGPA_Category"] = pd.cut(
        df["CGPA"],
        bins=[0, 6, 8, 10],
        labels=["low", "medium", "high"],
        ordered=True
    )

    ordinal_cols = ["CGPA_Category"]
    categories_order = [["low", "medium", "high"]]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"]
    )

    encoder = OrdinalEncoder(
        categories=categories_order,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    train_df_encoded[ordinal_cols] = encoder.fit_transform(train_df[ordinal_cols])
    test_df_encoded[ordinal_cols] = encoder.transform(test_df[ordinal_cols])

    print(train_df_encoded[ordinal_cols].head())
    print("=" * 80)
    return train_df_encoded, test_df_encoded


def run_label_encoding(data_path: str):
    """Executes Label Encoding on categorical stream targets."""
    print("Executing Label Encoding for Branch/Stream...")
    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"]
    )

    encoder = LabelEncoder()
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    train_df_encoded["Stream"] = encoder.fit_transform(train_df["Stream"])
    test_df_encoded["Stream"] = encoder.transform(test_df["Stream"])

    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("Stream Classes Mapped:", class_mapping)
    print(train_df_encoded["Stream"].head())
    print("=" * 80)
    return train_df_encoded, test_df_encoded


def run_simple_logistic_regression(df: pd.DataFrame):
    """Trains a Simple Logistic Regression baseline model on CGPA."""
    print("Executing Simple Logistic Regression Baseline...")
    X = df[["CGPA"]]
    y = df["PlacementStatus"]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_X, train_y)

    theta0 = model.intercept_[0]
    theta1 = model.coef_[0][0]

    print(f"Intercept (theta0): {theta0:.4f}")
    print(f"Coefficient (theta1): {theta1:.4f}")

    y_pred = model.predict(test_X)
    print(f"Logistic Regression Accuracy: {accuracy_score(test_y, y_pred):.4f}")
    print("=" * 80)


# =================================================================================
# STEPS 2 TO 6: Decision Tree, Probability Scores, ROC-AUC, Visualization & Workflow
# =================================================================================

def run_decision_tree_roc_auc_pipeline(data_path: str):
    """
    Executes Steps 2 through 6 of the ROC-AUC workflow.
    """
    # Step 1: Load and scale data
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data(data_path)

    # Step 2: Train Decision Tree Classifier
    print("Step 2: Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train)

    # Step 3: Predict probabilities for ROC-AUC
    print("Step 3: Calculating class probability estimates (predict_proba)...")
    y_prob = dt_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = dt_model.predict(X_test_scaled)

    # Step 4: Construct ROC Curve & Calculate AUC Score
    print("Step 4: Computing FPR, TPR, and ROC-AUC Score...")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- MODEL METRICS ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # AUC Quality Assessment
    if roc_auc >= 0.9:
        discrimination = "Excellent"
    elif roc_auc >= 0.8:
        discrimination = "Good"
    elif roc_auc >= 0.7:
        discrimination = "Fair"
    elif roc_auc >= 0.6:
        discrimination = "Poor"
    else:
        discrimination = "No better than random"
    print(f"Discrimination Ability Rating: {discrimination}")

    # Step 5: ROC Curve Visualization Plot
    print("\nStep 5: Displaying ROC Curve Plot...")
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Decision Tree (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Placement Prediction")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Step 6: Complete Workflow Summary Evaluation
    print("=" * 80)
    print("Step 6: Complete Workflow Summary & Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Execution Flow
    main_df = pd.read_csv(DATASET_PATH)

    explore_dataset(main_df)
    perform_statistical_analysis(main_df)

    run_one_hot_encoding(DATASET_PATH)
    run_ordinal_encoding(DATASET_PATH)
    run_label_encoding(DATASET_PATH)

    run_simple_logistic_regression(main_df)

    # Run full Decision Tree + ROC-AUC Pipeline (Steps 1 to 6)
    run_decision_tree_roc_auc_pipeline(DATASET_PATH)