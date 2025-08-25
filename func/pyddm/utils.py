import pandas as pd
import matplotlib.pyplot as plt

# Functionality for data pipeline


# Loads the data
def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path).dropna()
    return df


# Prepares subset based on specified dimension and category
def prepare_data(df, dimension, category):
    return df[df[dimension] == category].copy()


# Just in case the values in the choice column are not 0 and 1
def map_choices_to_binary(df, choice_col):
    unique_vals = df[choice_col].dropna().unique()

    if set(unique_vals).issubset({0, 1, True, False}):
        return df  # Already binary

    elif set(unique_vals) == {"yes", "no"
                              } or set(unique_vals) == {"no", "yes"}:
        df[choice_col] = df[choice_col].map({"yes": 1, "no": 0})

    elif set(unique_vals) == {"Yes", "No"
                              } or set(unique_vals) == {"No", "Yes"}:
        df[choice_col] = df[choice_col].map({"Yes": 1, "No": 0})

    else:
        raise ValueError(
            f"Unrecognized choice values in column '{choice_col}': {unique_vals}"
        )

    return df


# Compute drift rates by binning participants by questionnaire score
def compute_drift_rates_per_participant(df, participant_col, rt_col,
                                        choice_col, dimension_cols,
                                        anxiety_measures, depression_measures,
                                        drift_model_creator, drift_fit_func):
    results = []

    for dimension in dimension_cols:
        print(f"[INFO] Processing dimension: {dimension}")

        if dimension == "Circumplex":
            if "Affiliation" not in df.columns and "Dominance" not in df.columns:
                df["Affiliation"] = df["Circumplex"].where(
                    df["Circumplex"].str.contains("affiliation", case=False))
                df["Dominance"] = df["Circumplex"].where(
                    df["Circumplex"].str.contains("dominance", case=False))
            for circ_type, col in [("Affiliation", "Affiliation"),
                                   ("Dominance", "Dominance")]:
                for category in df[col].dropna().unique():
                    print(f"  ↳ {circ_type} Category: {category}")
                    subset = df[df[col] == category].copy()

                    for measure_set, measure_type in [
                        (anxiety_measures, 'Anxiety'),
                        (depression_measures, 'Depression')
                    ]:
                        for measure in measure_set:
                            print(
                                f"    ↳ Fitting per-person model for {measure_type} measure: {measure}"
                            )

                            scores = df[[participant_col,
                                         measure]].drop_duplicates().dropna()
                            scores.columns = [participant_col, 'score']
                            subset_scored = subset.merge(scores,
                                                         on=participant_col)

                            if subset_scored['score'].nunique() < 3:
                                print(
                                    f"    [SKIP] Not enough unique scores for {measure}"
                                )
                                continue

                            for pid, person_df in subset_scored.groupby(
                                    participant_col):
                                if len(person_df) < 5:
                                    continue

                                model = drift_model_creator()
                                try:
                                    fit_result = drift_fit_func(
                                        model, person_df[[
                                            rt_col, choice_col, 'score'
                                        ]], rt_col, choice_col, 'score')
                                    if fit_result is None:
                                        continue

                                    drift_value = float(
                                        fit_result.parameters()["drift"]
                                        ["drift"])
                                    alpha_value = float(
                                        fit_result.parameters()["bound"]["B"])
                                    bias_value = float(
                                        fit_result.parameters()["IC"]["x0"])

                                    results.append({
                                        'Participant':
                                        pid,
                                        'Dimension':
                                        circ_type if dimension == "Circumplex"
                                        else dimension,
                                        'Category':
                                        category,
                                        'Measure':
                                        measure,
                                        'Type':
                                        measure_type,
                                        'Score':
                                        person_df['score'].iloc[0],
                                        'DriftRate':
                                        drift_value,
                                        'Alpha':
                                        alpha_value,
                                        'Bias':
                                        bias_value
                                    })
                                except Exception as e:
                                    print(
                                        f"    [ERROR] Fit failed for participant {pid}: {e}"
                                    )
                                    continue

        else:
            # Handle Valence as before
            for category in df[dimension].dropna().unique():
                print(f"  ↳ Category: {category}")
                subset = df[df[dimension] == category].copy()

                for measure_set, measure_type in [
                    (anxiety_measures, 'Anxiety'),
                    (depression_measures, 'Depression')
                ]:
                    for measure in measure_set:
                        print(
                            f"    ↳ Fitting per-person model for {measure_type} measure: {measure}"
                        )

                        scores = df[[participant_col,
                                     measure]].drop_duplicates().dropna()
                        scores.columns = [participant_col, 'score']
                        subset_scored = subset.merge(scores,
                                                     on=participant_col)

                        if subset_scored['score'].nunique() < 3:
                            print(
                                f"    [SKIP] Not enough unique scores for {measure}"
                            )
                            continue

                        for pid, person_df in subset_scored.groupby(
                                participant_col):
                            if len(person_df) < 5:
                                continue

                            model = drift_model_creator()
                            try:
                                fit_result = drift_fit_func(
                                    model,
                                    person_df[[rt_col, choice_col, 'score']],
                                    rt_col, choice_col, 'score')
                                if fit_result is None:
                                    continue

                                drift_value = float(
                                    fit_result.parameters()["drift"]["drift"])
                                alpha_value = float(
                                    fit_result.parameters()["bound"]["B"])
                                bias_value = float(
                                    fit_result.parameters()["IC"]["x0"])

                                results.append({
                                    'Participant':
                                    pid,
                                    'Dimension':
                                    circ_type if dimension == "Circumplex" else
                                    dimension,
                                    'Category':
                                    category,
                                    'Measure':
                                    measure,
                                    'Type':
                                    measure_type,
                                    'Score':
                                    person_df['score'].iloc[0],
                                    'DriftRate':
                                    drift_value,
                                    'Alpha':
                                    alpha_value,
                                    'Bias':
                                    bias_value
                                })
                            except Exception as e:
                                print(
                                    f"    [ERROR] Fit failed for participant {pid}: {e}"
                                )
                                continue

    return pd.DataFrame(results,
                        columns=[
                            'Participant', 'Dimension', 'Category', 'Measure',
                            'Type', 'Score', 'DriftRate', 'Alpha', 'Bias'
                        ])


# Plots the graphs to showcase the drift rate over score
def plot_drift_rates(results_df, dimension, anxiety_measures,
                     depression_measures):
    plt.figure(figsize=(10, 6))
    subset_df = results_df[results_df['Dimension'] == dimension]

    for measure in subset_df['Measure'].unique():
        measure_df = subset_df[subset_df['Measure'] == measure]
        if not measure_df.empty:
            color = 'purple' if measure in anxiety_measures else 'red'

            # Sort for smooth plotting
            measure_df = measure_df.sort_values(by="Score")

            plt.plot(measure_df['Score'],
                     measure_df['DriftRate'],
                     marker='o',
                     linestyle='-',
                     label=measure,
                     color=color)

    plt.xlabel("Questionnaire Score")
    plt.ylabel("Drift Rate")
    plt.title(f"Drift Rate by Score (Per Participant): {dimension}")
    plt.legend(loc='best', frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
