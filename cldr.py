"""
Member A - Data Cleaning + Dimensionality Reduction (Objective 2)
Heart Disease Prediction Project

Pipeline: Load -> Clean -> Split -> Baseline -> RFE sweep -> LDA -> Compare -> Save results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG — update these to match your actual datasets
# ============================================================
DATA_PATH_1 = "data/diabetes_prediction_dataset.csv"      # path to your first dataset file
DATA_PATH_2 = "data/cardio_train.csv"                     # path to your second dataset file
TARGET_COLUMN = "diabetes"                                # name of the label column
CATEGORICAL_COLUMNS = ["gender","smoking_history"]        # list any text/categorical columns here
RANDOM_STATE = 42                                         # SHARED seed
TEST_SIZE = 0.2
RFE_FEATURE_COUNTS = [4, 6, 8]                            # feature counts to sweep for RFE
OUTPUT_RESULTS_CSV = "objective2_reduction_results.csv"


def load_and_inspect(path):
    # Detect delimiter automatically (cardio_train often uses ';')
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(path)
        
    print("=" * 60)
    print(f"LOADING FILE: {path}")
    print("SHAPE:", df.shape)
    print("\nDTYPES:\n", df.dtypes)
    print("=" * 60)
    return df


def clean(df):
    # 1. Clean up column names by stripping any accidental hidden spaces
    df.columns = df.columns.str.strip()
    
    # 2. Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # 3. Force conversion of text columns to numbers regardless of their dtype
    for col in CATEGORICAL_COLUMNS:
        col = col.strip()  # Remove spaces from config names too
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df


def split_data(df):
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds)


def run_baseline(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    acc, f1 = evaluate(model, X_train, y_train, X_test, y_test)
    print(f"\nBASELINE (all {X_train.shape[1]} features): acc={acc:.4f}, f1={f1:.4f}")
    return acc, f1


def run_rfe_sweep(X_train, X_test, y_train, y_test, baseline_acc):
    results = []
    for n in RFE_FEATURE_COUNTS:
        if n >= X_train.shape[1]:
            continue
        selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=n)
        selector.fit(X_train, y_train)
        X_tr_r = selector.transform(X_train)
        X_te_r = selector.transform(X_test)

        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        acc, f1 = evaluate(model, X_tr_r, y_train, X_te_r, y_test)
        retained = (acc / baseline_acc) * 100
        selected_features = X_train.columns[selector.support_].tolist()

        print(f"RFE n={n}: acc={acc:.4f}, f1={f1:.4f}, retained={retained:.1f}%")
        print(f"  Selected features: {selected_features}")

        results.append({
            "technique": f"RFE_n{n}",
            "n_features": n,
            "accuracy": acc,
            "f1_score": f1,
            "pct_baseline_retained": retained,
            "selected_features": ", ".join(selected_features),
        })
    return results


def run_lda(X_train, X_test, y_train, y_test, baseline_acc):
    lda = LinearDiscriminantAnalysis(n_components=1)  # max components = n_classes - 1 for binary
    X_tr_lda = lda.fit_transform(X_train, y_train)
    X_te_lda = lda.transform(X_test)

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    acc, f1 = evaluate(model, X_tr_lda, y_train, X_te_lda, y_test)
    retained = (acc / baseline_acc) * 100

    print(f"LDA (1 component): acc={acc:.4f}, f1={f1:.4f}, retained={retained:.1f}%")

    return {
        "technique": "LDA",
        "n_features": 1,
        "accuracy": acc,
        "f1_score": f1,
        "pct_baseline_retained": retained,
        "selected_features": "N/A (transformed component)",
    }


def main():
    # 1. Load the original Diabetes dataset
    print("Loading primary diabetes dataset...")
    df_diabetes = load_and_inspect(DATA_PATH_1)
    df_diabetes = clean(df_diabetes)
    
    # 2. Load the secondary Cardiovascular dataset
    print("\nLoading cardio_train dataset...")
    df_cardio = load_and_inspect(DATA_PATH_2)
    df_cardio.columns = df_cardio.columns.str.strip()
    
    print("Processing and aligning cardio data...")
    # Calculate BMI from height (cm) and weight (kg) to map features correctly
    df_cardio['bmi'] = df_cardio['weight'] / ((df_cardio['height'] / 100) ** 2)
    
    # Map cardio gender indicators (1=women, 2=men) to standard binary labels (0, 1)
    df_cardio['gender_encoded'] = df_cardio['gender'].map({1: 0, 2: 1}).fillna(0).astype(int)
    
    # Standard text formatting for categorical maps
    df_cardio['smoking_history_encoded'] = df_cardio['smoke'].map({0: 0, 1: 1}).fillna(0).astype(int)
    
    # Isolate relevant alignment properties
    X_cardio_features = pd.DataFrame({
        'gender': df_cardio['gender_encoded'],
        'age': df_cardio['age'] / 365.25, # Transform age from days to years
        'hypertension': df_cardio['ap_hi'].apply(lambda x: 1 if x >= 140 else 0),
        'heart_disease': df_cardio['cardio'], # Set cardio status to feature tracking
        'smoking_history': df_cardio['smoking_history_encoded'],
        'bmi': df_cardio['bmi'],
        'HbA1c_level': 5.5, # Standard median filling baseline indicator
        'blood_glucose_level': df_cardio['gluc'].map({1: 90, 2: 125, 3: 160}).fillna(90)
    })
    
    # Standard health cross-analysis tracks diabetes predictions based on glucose level class
    X_cardio_features[TARGET_COLUMN] = df_cardio['gluc'].apply(lambda x: 1 if x == 3 else 0)
    
    # 3. Concatenate both clean datasets safely
    print("\nMerging datasets...")
    df = pd.concat([df_diabetes, X_cardio_features], axis=0, ignore_index=True)
    print(f"Final Integrated Matrix Shape: {df.shape}")
    
    # 4. Proceed with train/test splits and training loops
    X_train, X_test, y_train, y_test = split_data(df)
    baseline_acc, baseline_f1 = run_baseline(X_train, X_test, y_train, y_test)

    all_results = [{
        "technique": "Baseline_all_features",
        "n_features": X_train.shape[1],
        "accuracy": baseline_acc,
        "f1_score": baseline_f1,
        "pct_baseline_retained": 100.0,
        "selected_features": ", ".join(X_train.columns.tolist()),
    }]

    all_results += run_rfe_sweep(X_train, X_test, y_train, y_test, baseline_acc)
    all_results.append(run_lda(X_train, X_test, y_train, y_test, baseline_acc))

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("pct_baseline_retained", ascending=False)

    print("\n" + "=" * 60)
    print("FINAL COMBINED COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"\nResults metrics saved to {OUTPUT_RESULTS_CSV}")

    # 5. Overwrite and export the true transformed array metrics
    print("\nSaving combined processed datasets...")
    df.to_csv("data/diabetes_cleaned_all_features.csv", index=False)
    print("Saved: data/diabetes_cleaned_all_features.csv")
    
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_full = df.drop(TARGET_COLUMN, axis=1)
    y_full = df[TARGET_COLUMN]
    X_lda_transformed = lda.fit_transform(X_full, y_full)
    
    lda_df = pd.DataFrame(X_lda_transformed, columns=['LDA_Component_1'])
    lda_df[TARGET_COLUMN] = y_full
    
    lda_df.to_csv("data/diabetes_reduced_lda.csv", index=False)
    print("Saved: data/diabetes_reduced_lda.csv")
    print("=" * 60)
    print("Done! Hand these updated data files off to Member B and Member C.")


if __name__ == "__main__":
    main()
