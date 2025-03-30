
################
# Dependencies #
################

from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayUniformMixture, ICUniform  # type: ignore
from config import drift_model_config
import os
import contextlib
import sys
import logging
logging.getLogger("pyddm").setLevel(logging.WARNING)

############
# Cleaning #
############

# Cleaning the NA values from the data 

def na_cleanout(data):
    data = data.dropna()
    return data

# Cleaning the outputs from the model (making sure there are none hahah)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

####################
# Model Definition #
####################

# Defining the model based on configs defined in config.py file

def create_drift_model(cfg=None):
    if cfg is None:
        cfg = drift_model_config()

    model = Model(
        drift=DriftConstant(
            drift=Fittable(minval=cfg['drift']['minval'], maxval=cfg['drift']['maxval'])
        ),
        noise=NoiseConstant(noise=cfg['noise']),
        bound=BoundConstant(
            B=Fittable(minval=cfg['bound']['minval'], maxval=cfg['bound']['maxval'])
        ),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(
                nondectime=Fittable(
                    minval=cfg['nondectime']['minval'],
                    maxval=cfg['nondectime']['maxval']
                )
            ),
            OverlayUniformMixture(
                umixturecoef=Fittable(
                    minval=cfg['umixturecoef']['minval'],
                    maxval=cfg['umixturecoef']['maxval']
                )
            )
        ]),
        IC=ICUniform(),
        dt=cfg['dt'],
        T_dur=cfg['T_dur']
    )

    return model

def fit_model(model, data, rt_column, choice_column):
    try:
        sample = Sample.from_pandas_dataframe(data, rt_column_name=rt_column, choice_column_name=choice_column)
        with suppress_stdout_stderr():  # suppress optimizer output
            fit_result = model.fit(sample)
        return fit_result
    except Exception as e:
        print(f"Fit failed for this subset: {e}")
        return None