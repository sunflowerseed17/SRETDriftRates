#################
# Dependencies  #
#################




####################################################
# Configs for word categorization and drift model  #
####################################################


def get_columns(data):
    return data.columns.tolist()

# Dynamically extract words and their grouping from specified columns
def get_words(data, word_column, affiliation_column, dominance_column):
    return dict(zip(data[word_column], zip(data[affiliation_column], data[dominance_column])))

# Dynamically extract decisions and RTs
def get_decisions_and_RT(data, decision_column, RT_column):
    return dict(zip(data[decision_column], data[RT_column]))

# Dynamically extract scores for any given measures
def get_scores(data, participant_column, measure_columns):
    scores = data.groupby(participant_column)[measure_columns].first()
    return scores.to_dict(orient='index')

# Drift model configuration defaults 
def drift_model_config():
    return {
        'drift': {'minval': -10, 'maxval': 10},
        'noise': 1,
        'bound': {'minval': 0.5, 'maxval': 4.0},
        'nondectime': {'minval': 0, 'maxval': 1.0},
        'umixturecoef': {'minval': 0, 'maxval': 0.1},
        'dt': 0.01,
        'T_dur': 8.0
    }

