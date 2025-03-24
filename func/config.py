###################################
# Configs for word categorization #
###################################

# Extracting the columns names from the data

def get_columns(data):
    columns = data.columns.tolist()
    return columns

# Extracting the words from the data and their groupings by affiliation and dominance

def get_words(data, word_column, affiliation_column, dominance_column):
    words = data[word_column].tolist()
    affiliations = data[affiliation_column].tolist()
    dominances = data[dominance_column].tolist()
    word_aff_doms = {}
    for i in range(len(words)):
        word_aff_doms[words[i]] = (affiliations[i], dominances[i])
    return word_aff_doms

def get_decisions_and_RT(data, decision_column, RT_column):
    decision_RT = {}
    decisions = data[decision_column].tolist()
    RTs = data[RT_column].tolist()
    for i in range(len(decisions)):
        decision_RT[decisions[i]] = RTs[i]
    return decision_RT

#####################################
# Configs for clinical distribution #
#####################################

# Extracting the scores for each participants for depression-related measures

def get_depression_scores(data, participant_column, *args):
    participant_depression_scores = {}
    unique_participants = data[participant_column].unique()
    for participant in unique_participants:
        first_idx = data[data[participant_column] == participant].index[0]
        scores = [data[arg][first_idx] for arg in args]
        participant_depression_scores[participant] = scores
    return participant_depression_scores

# Extracting the scores for each participants for anxiety-related measures

def get_anxiety_scores(data, participant_column, *args):
    participant_anxiety_scores = {}
    unique_participants = data[participant_column].unique()
    for participant in unique_participants:
        first_idx = data[data[participant_column] == participant].index[0]
        scores = [data[arg][first_idx] for arg in args]
        participant_anxiety_scores[participant] = scores
    return participant_anxiety_scores


###########################
# Configs for drift model #
###########################

drift_model_config = {
    'drift': {
        'type': 'DriftConstant',
        'params': {
            'drift': {
                'fittable': True,
                'minval': -10,
                'maxval': 10
            }
        }
    },
    'noise': {
        'type': 'NoiseConstant',
        'params': {
            'noise': 1  
        }
    },
    'bound': {
        'type': 'BoundConstant',
        'params': {
            'B': {
                'fittable': True,
                'minval': 0.5,
                'maxval': 4.0
            }
        }
    },
    'overlay': {
        'type': 'OverlayChain',
        'overlays': [
            {
                'type': 'OverlayNonDecision',
                'params': {
                    'nondectime': {
                        'fittable': True,
                        'minval': 0,
                        'maxval': 1.0
                    }
                }
            },
            {
                'type': 'OverlayUniformMixture',
                'params': {
                    'umixturecoef': {
                        'fittable': True,
                        'minval': 0,
                        'maxval': 0.1
                    }
                }
            }
        ]
    },
    'IC': {
        'type': 'ICUniform'
    },
    'dt': 0.01,
    'T_dur': 8.0
}


