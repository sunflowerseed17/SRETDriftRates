################
# Dependencies #
################

import numpy as np
import pandas as pd

################################################################
# Hard coding ddm because the packaged have failed me here haha #
################################################################


def simulate_ddm_trials(v, a=1.0, t0=0.3, s=1.0, dt=0.001, n_trials=1000, max_t=10):
    rts, choices = [], []
    for _ in range(n_trials):
        x = 0
        t = 0
        while abs(x) < a and t < max_t:
            dx = v * dt + np.random.normal(scale=s*np.sqrt(dt))
            x += dx
            t += dt
        rt = t0 + t
        choice = int(x > 0)  # 1 = upper boundary (e.g., "yes"), 0 = lower boundary ("no")
        rts.append(rt)
        choices.append(choice)
    return pd.DataFrame({"rt": rts, "choice": choices})

################
# Parameters  #
################

wang_params = {
    "positive_high_dominance": {"v": 1.0, "t0": 0.30},
    "positive_high_affiliation": {"v": 1.0, "t0": 0.30},
    "negative_low_dominance": {"v": -0.6, "t0": 0.36},
    "negative_low_affiliation": {"v": -0.6, "t0": 0.36}
}

condition_counts = {k: 684 for k in wang_params}

################
# Simulation   #
################

df_all = []
for cond, params in wang_params.items():
    df = simulate_ddm_trials(v=params["v"], t0=params["t0"], n_trials=condition_counts[cond])
    df["condition"] = cond
    df_all.append(df)

df_sim = pd.concat(df_all, ignore_index=True)
df_sim.to_csv("./data/synthetic_ddm_wang2024.csv", index=False)
print(df_sim.head())