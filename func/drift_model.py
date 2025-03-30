
################
# Dependencies #
################

from pyddm import Model, Fittable, Sample
from pyddm.models import Drift, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayUniformMixture, ICUniform  # type: ignore
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

class DriftScoreDependent(Drift):
    name = "Drift varies with questionnaire score"
    required_parameters = ["a", "b"]
    required_conditions = ["score"]

    def get_drift(self, x, conditions, **kwargs):
        return self.a * conditions["score"] + self.b

def create_drift_model(cfg=None):
    if cfg is None:
        cfg = drift_model_config()

    return Model(
        drift=DriftScoreDependent(
            a=Fittable(minval=-5, maxval=5),
            b=Fittable(minval=-5, maxval=5)
        ),
        noise=NoiseConstant(noise=cfg['noise']),
        bound=BoundConstant(B=Fittable(minval=cfg['bound']['minval'], maxval=cfg['bound']['maxval'])),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=cfg['nondectime']['minval'], maxval=cfg['nondectime']['maxval'])),
            OverlayUniformMixture(umixturecoef=Fittable(minval=cfg['umixturecoef']['minval'], maxval=cfg['umixturecoef']['maxval']))
        ]),
        IC=ICUniform(),
        dt=cfg['dt'],
        T_dur=cfg['T_dur']
    )

def fit_model(model, data, rt_column, choice_column, score_column):
    df = data.rename(columns={
        rt_column: 'rt',
        choice_column: 'choice',
        score_column: 'score'
    })

    sample = Sample.from_pandas_dataframe(
        df[["rt", "choice"]],
        rt_column_name='rt',
        choice_column_name='choice'
    )
    sample.conditions["score"] = df["score"].values

    with suppress_stdout_stderr():
        return fit_adjust_model(sample=sample, model=model)
