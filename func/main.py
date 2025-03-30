################
# Dependencies #
################


from utils import map_choices_to_binary,load_and_prepare_data, compute_drift_rates, plot_drift_rates
from drift_model import create_drift_model, fit_model
import logging
logging.getLogger("pyddm").setLevel(logging.ERROR)

#########################################
# Running the model on specific dataset #
#########################################


data_path = input("Enter the dataset path (e.g., data/SRET2019.csv): ")
participant_col = input("Enter the participant ID column name(e.g., Subject): ")
rt_col = input("Enter the response time (RT) column name (e.g., SERT.RT): ")
choice_col = input("Enter the choice/response column name (e.g., Response): ")
dimension_cols = input("Enter dimension columns (comma-separated, e.g., Valence,Circumplex): ").split(',')
anxiety_measures = input("Enter anxiety measure column names (comma-separated, e.g., LSAS,SPIN,BFNE,STAI-S,STAI-T,FPES) : ").split(',')
depression_measures = input("Enter depression measure column names (comma-separated, e.g., BDI,RSES): ").split(',')

# Load data

df = load_and_prepare_data(data_path)
df = map_choices_to_binary(df, choice_col)
df[rt_col] = df[rt_col] / 1000 # Convert RT from ms to seconds


# Compute drift rates
results_df = compute_drift_rates(
    df, participant_col, rt_col, choice_col,
    dimension_cols, anxiety_measures, depression_measures,
    create_drift_model, fit_model
)

# Save results to CSV
results_df.to_csv('output/drift_rates.csv', index=False)

# Generate plots
for dimension in dimension_cols:
    plot_drift_rates(results_df, dimension, anxiety_measures, depression_measures)
