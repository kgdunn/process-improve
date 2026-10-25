"""PCA of the LDPE process data: which batch the model does not explain, and in which variables.

The data are 54 batches from a tubular low-density polyethylene reactor, bundled with the package:
14 process variables (temperatures, flows and pressure, in the reactor's two zones) and 5 quality
variables. Run it as a script; each figure opens in the browser.
"""

# %%
import pathlib

import pandas as pd

import process_improve
from process_improve.multivariate.methods import PCA, MCUVScaler

csv = pathlib.Path(process_improve.__file__).parent / "datasets" / "multivariate" / "LDPE" / "LDPE.csv"
process = pd.read_csv(csv, index_col=0).loc[:, "Tin":"Press"]

# %% Scale each variable to mean 0 and variance 1, then fit three components.
scaled = MCUVScaler().fit_transform(process)
pca = PCA(n_components=3).fit(scaled)
print(pca.eigenvalue_summary())

# %% The fitted model: a score per batch and a loading per variable, for each component.
print(pca.scores_.head())
print(pca.loadings_.round(2))

# %% SPE is the distance from a batch to the model; Hotelling's T2 is its distance along the model from the center.
# Each has a column per component: the last is the value with all three.
spe = pca.spe_.iloc[:, -1]
spe_limit = pca.spe_limit(conf_level=0.95)
t2 = pca.hotellings_t2_.iloc[:, -1]
t2_limit = pca.hotellings_t2_limit(conf_level=0.95)
print(f"SPE above its 95% limit of {spe_limit:.2f}: batches {list(spe.index[spe > spe_limit])}")
print(f"T2 above its 95% limit of {t2_limit:.2f}: batches {list(t2.index[t2 > t2_limit])}")
print("SPE of the last four batches:", spe.tail(4).round(2).tolist())

# %% The SPE rises over the last four batches. Which variables does the model not explain in the last one?
last = spe.index[-1]
contributions = pca.spe_contributions(scaled).loc[last]
print(f"Largest contributions to the SPE of batch {last}:")
print(contributions.sort_values(key=abs, ascending=False).head(4).round(2))

# %% The plots
pca.score_plot(pc_horiz=1, pc_vert=2).show()
pca.loading_plot(pc_horiz=1, pc_vert=2).show()
pca.spe_plot().show()
pca.t2_plot().show()
