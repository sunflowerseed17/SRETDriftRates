
################
# Dependencies #
################


import pandas as pd


###################################
# Functionality for data pipeline #
###################################


def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path).dropna()
    return df

# Prepare subset based on specified dimension and category
def prepare_data(df, dimension, category):
    return df[df[dimension] == category].copy()

# Dynamically obtain participant scores
def get_participant_scores(df, participant_col, anxiety_measures, depression_measures):
    anxiety_scores = df.groupby(participant_col)[anxiety_measures].first()
    depression_scores = df.groupby(participant_col)[depression_measures].first()
    return anxiety_scores, depression_scores

def map_choices_to_binary(df, choice_col):
    """
    Converts any 'yes'/'no'-like responses to binary (1/0).
    If already binary, it passes unchanged.
    """
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

def compute_drift_rates(df, participant_col, rt_col, choice_col,
                        dimension_cols, anxiety_measures, depression_measures,
                        drift_model_creator, drift_fit_func):
    results = []

    anxiety_scores, depression_scores = get_participant_scores(
        df, participant_col, anxiety_measures, depression_measures
    )

    for dimension in dimension_cols:
        for category in df[dimension].dropna().unique():
            subset = prepare_data(df, dimension, category)

            for measure_set, measure_type, scores in [
                (anxiety_measures, 'Anxiety', anxiety_scores),
                (depression_measures, 'Depression', depression_scores)
            ]:
                for measure in measure_set:
                    subset = subset.copy()
                    subset['score'] = subset[participant_col].map(scores[measure])

                    try:
                        drift_model = drift_model_creator()
                        fit_result = drift_fit_func(
                            drift_model,
                            subset[[rt_col, choice_col]].rename(columns={rt_col: 'RT', choice_col: 'Response'}),
                            'RT',
                            'Response'
                        )

                        if fit_result is None:
                            print(f"[WARN] Fit failed for {dimension}-{category}-{measure}")
                            continue

                        drift_rate = fit_result.parameters()['drift']
                        results.append((dimension, category, measure, measure_type, drift_rate))

                    except Exception as e:
                        print(f"[ERROR] Failed for {dimension}-{category}-{measure}: {e}")
                        continue

    results_df = pd.DataFrame(results, columns=['Dimension', 'Category', 'Measure', 'Type', 'DriftRate'])
    return results_df

def plot_drift_rates(results_df, dimension, anxiety_measures, depression_measures):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    subset_df = results_df[results_df['Dimension'] == dimension]

    for measure in subset_df['Measure'].unique():
        measure_df = subset_df[subset_df['Measure'] == measure]
        if not measure_df.empty:
            type_color = 'purple' if measure in anxiety_measures else 'red'
            plt.plot(measure_df['Category'], measure_df['DriftRate'], marker='o', label=measure, color=type_color)

    plt.xlabel("Questionnaire Score")
    plt.ylabel('Drift Rate')
    plt.title(f'Drift Rate by Questionnaire Score: {dimension}')
    plt.legend(loc='best', frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
