
################
# Dependencies #
################

from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayUniformMixture, ICUniform  # type: ignore
from pyddm.functions import fit_adjust_model
from config import drift_model_config
import os
import contextlib
import sys
import logging

# Suppress pyddm debug logging and output
logging.getLogger("pyddm").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

############
# Cleaning #
############

def na_cleanout(data):
    return data.dropna()

# Suppress stdout/stderr context
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

def create_drift_model(cfg=None):
    if cfg is None:
        cfg = drift_model_config()

    return Model(
        drift=DriftConstant(drift=Fittable(minval=cfg['drift']['minval'], maxval=cfg['drift']['maxval'])),
        noise=NoiseConstant(noise=cfg['noise']),
        bound=BoundConstant(B=Fittable(minval=cfg['bound']['minval'], maxval=cfg['bound']['maxval'])),
        overlay=OverlayNonDecision(nondectime=cfg['nondectime']),  
        IC=ICUniform(),  
        dt=cfg['dt'],
        T_dur=cfg['T_dur']
    )

def fit_model(model, data, rt_column, choice_column, score_column):
    sample = Sample.from_pandas_dataframe(
        data.rename(columns={
            rt_column: 'rt',
            choice_column: 'choice'
        }),
        rt_column_name='rt',
        choice_column_name='choice'
    )
    with suppress_stdout_stderr():
        return fit_adjust_model(sample=sample, model=model)
