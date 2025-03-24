
################
# Dependencies #
################

from config import drift_model_config
from ddm import Model, Fittable
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayUniformMixture, ICUniform # type: ignore




############
# Cleaning #
############

# Cleaning the NA values from the data 

def na_cleanout(data):
    data = data.dropna()
    return data

####################
# Model Definition #
####################

# Setting the name of columns based on the initial input

def column_names(data):
    


# Defining the model based on configs defined in config.py file

def drm():
    cfg = drift_model_config  # for shorter reference

    model = Model(
        drift=DriftConstant(
            drift=Fittable(
                minval=cfg['drift']['params']['drift']['minval'],
                maxval=cfg['drift']['params']['drift']['maxval']
            )
        ),
        noise=NoiseConstant(
            noise=cfg['noise']['params']['noise']
        ),
        bound=BoundConstant(
            B=Fittable(
                minval=cfg['bound']['params']['B']['minval'],
                maxval=cfg['bound']['params']['B']['maxval']
            )
        ),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(
                nondectime=Fittable(
                    minval=cfg['overlay']['overlays'][0]['params']['nondectime']['minval'],
                    maxval=cfg['overlay']['overlays'][0]['params']['nondectime']['maxval']
                )
            ),
            OverlayUniformMixture(
                umixturecoef=Fittable(
                    minval=cfg['overlay']['overlays'][1]['params']['umixturecoef']['minval'],
                    maxval=cfg['overlay']['overlays'][1]['params']['umixturecoef']['maxval']
                )
            )
        ]),
        IC=ICUniform(),
        dt=cfg['dt'],
        T_dur=cfg['T_dur']
    )

    return model

def fit_model(model, data):
    sample = Sample.from_pandas_dataframe(data, rt_column_name='decision_column', choice_column_name='Response')


