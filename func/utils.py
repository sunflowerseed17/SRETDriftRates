
################
# Dependencies #
################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


###################################
# Functionality for data pipeline #
###################################

# Loads the data
def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path).dropna()
    return df

# Prepares subset based on specified dimension and category
def prepare_data(df, dimension, category):
    return df[df[dimension] == category].copy()

# Obtains participant scores
def get_participant_scores(df, participant_col, anxiety_measures, depression_measures):
    anxiety_scores = df.groupby(participant_col)[anxiety_measures].first()
    depression_scores = df.groupby(participant_col)[depression_measures].first()
    return anxiety_scores, depression_scores

# Just in case the values in the choice column are not 0 and 1 
def map_choices_to_binary(df, choice_col):
    unique_vals = df[choice_col].dropna().unique()
    
    if set(unique_vals).issubset({0, 1, True, False}):
        return df  # Already binary

    elif set(unique_vals) == {"yes", "no"} or set(unique_vals) == {"no", "yes"}:
        df[choice_col] = df[choice_col].map({"yes": 1, "no": 0})

    elif set(unique_vals) == {"Yes", "No"} or set(unique_vals) == {"No", "Yes"}:
        df[choice_col] = df[choice_col].map({"Yes": 1, "No": 0})

    else:
        raise ValueError(f"Unrecognized choice values in column '{choice_col}': {unique_vals}")

    return df

# Computes drift rates for each participant and dimension
def compute_drift_rates(df, participant_col, rt_col, choice_col,
                        dimension_cols, anxiety_measures, depression_measures,
                        drift_model_creator, drift_fit_func):
    results = []

    anxiety_scores, depression_scores = get_participant_scores(
        df, participant_col, anxiety_measures, depression_measures
    )

    for dimension in dimension_cols:
        print(f"[INFO] Processing dimension: {dimension}")
        for category in df[dimension].dropna().unique():
            print(f"  ↳ Category: {category}")
            subset = prepare_data(df, dimension, category)

            unique_participants = subset[participant_col].nunique()
            print(f"    [INFO] Found {unique_participants} unique participants")

            if unique_participants < 5:
                print(f"    [SKIP] Not enough participants for {dimension}-{category} (need ≥ 5)")
                continue

            for measure_set, measure_type, scores in [
                (anxiety_measures, 'Anxiety', anxiety_scores),
                (depression_measures, 'Depression', depression_scores)
            ]:
                for measure in measure_set:
                    print(f"    ↳ Fitting model for {measure_type} measure: {measure}")
                    subset = subset.copy()
                    try:
                        subset['score'] = subset[participant_col].map(scores[measure])
                        subset = subset.dropna(subset=['score', rt_col, choice_col])

                        score_variability = subset['score'].nunique()
                        if score_variability < 3:
                            print(f"    [SKIP] Not enough unique scores for {measure} (found {score_variability})")
                            continue

                        drift_model = drift_model_creator()

                        fit_result = drift_fit_func(
                            drift_model,
                            subset[[rt_col, choice_col, 'score']],
                            rt_col,
                            choice_col,
                            'score'
                        )

                        if fit_result is None:
                            print(f"    [WARN] Fit failed for {dimension}-{category}-{measure}")
                            continue

                        a = fit_result.parameters()['drift']['a']
                        b = fit_result.parameters()['drift']['b']
                        results.append((dimension, category, measure, measure_type, a, b))

                    except Exception as e:
                        print(f"    [ERROR] Failed for {dimension}-{category}-{measure}: {e}")
                        continue

    results_df = pd.DataFrame(results, columns=['Dimension', 'Category', 'Measure', 'Type', 'DriftSlope_a', 'DriftIntercept_b'])
    return results_df


# Plots the graphs to showcase the drift rate over score 
def plot_drift_rates(results_df, dimension, anxiety_measures, depression_measures):
    plt.figure(figsize=(10, 6))
    subset_df = results_df[results_df['Dimension'] == dimension]

    score_range = np.linspace(0, 100, 200)  # Assuming score ranges from 0–100

    for measure in subset_df['Measure'].unique():
        measure_df = subset_df[subset_df['Measure'] == measure]
        if not measure_df.empty:
            a = measure_df['DriftSlope_a'].iloc[0]
            b = measure_df['DriftIntercept_b'].iloc[0]
            drift_vals = a * score_range + b
            color = 'purple' if measure in anxiety_measures else 'red'
            plt.plot(score_range, drift_vals, label=measure, color=color)

    plt.xlabel("Questionnaire Score")
    plt.ylabel("Drift Rate")
    plt.title(f"Drift Rate by Questionnaire Score: {dimension}")
    plt.legend(loc='best', frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()