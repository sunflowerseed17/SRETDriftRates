################
# Dependencies #
################


from utils import map_choices_to_binary,load_and_prepare_data, compute_drift_rates_per_participant, plot_drift_rates
from drift_model import create_drift_model, fit_model
import logging
import os
logging.getLogger("pyddm").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

#########################################
# Running the model on specific dataset #
#########################################


use_defaults = input("Would you like to proceed with the initial config and dataset? Type 'no' if there is a new dataset or you want to customize: ").strip().lower()

if use_defaults in ['yes', 'y', '']:
    data_path = "data/SRET2019.csv"
    participant_col = "Subject"
    rt_col = "SERT.RT"
    choice_col = "Response"
    dimension_cols = ["Valence", "Circumplex"]
    anxiety_measures = ["LSAS", "SPIN", "BFNE", "STAI-S", "STAI-T", "FPES"]
    depression_measures = ["BDI", "RSES"]
else:
    data_path = input("Enter the dataset path (e.g., data/SRET2019.csv): ").strip()
    participant_col = input("Enter the participant ID column name (e.g., Subject): ").strip()
    rt_col = input("Enter the response time (RT) column name (e.g., SERT.RT): ").strip()
    choice_col = input("Enter the choice/response column name (e.g., Response): ").strip()
    dimension_cols = input("Enter dimension columns (comma-separated, e.g., Valence,Circumplex): ").strip().split(',')
    anxiety_measures = input("Enter anxiety measure column names (comma-separated, e.g., LSAS,SPIN,BFNE,STAI-S,STAI-T,FPES): ").strip().split(',')
    depression_measures = input("Enter depression measure column names (comma-separated, e.g., BDI,RSES): ").strip().split(',')

# Load data
df = load_and_prepare_data(data_path)
df = map_choices_to_binary(df, choice_col)
df[rt_col] = df[rt_col] / 1000 # Convert RT from ms to seconds


# Compute drift rates
results_df = compute_drift_rates_per_participant(
    df, participant_col, rt_col, choice_col,
    dimension_cols, anxiety_measures, depression_measures,
    create_drift_model, fit_model
)

# Save results to CSV
results_df.to_csv('output/drift_rates.csv', index=False)

# Generate plots
for dimension in dimension_cols:
    plot_drift_rates(results_df, dimension, anxiety_measures, depression_measures)
