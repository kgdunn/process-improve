# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.12 ('datamore')
#     language: python
#     name: python3
# ---

# +
import os
import pathlib
import sys

import pandas as pd

from process_improve.multivariate.methods import PCA, MCUVScaler

# +
while not str(basecwd := pathlib.Path.cwd()).lower().endswith("process-improve"):
    os.chdir(basecwd := pathlib.Path.cwd().parents[0])
sys.path.insert(0, str(basecwd))


# -
df = (
    pd.read_excel("notebooks_examples/multivariate/MyPCA.xlsx", sheet_name="Capsule Summary")
    .set_index("batch_for_analyze")
    .drop(["b_recipe"], axis=1)
    .iloc[0:31, 0:18]
)


scaler = MCUVScaler().fit(df)
mcuv = scaler.fit_transform(df)
A = 3
pca = PCA(n_components=A, missing_data_settings=dict(md_method="scp")).fit(mcuv)
spe_limit_95 = pca.spe_limit(conf_level=0.95)
print(pca.x_scores)
print(pca.x_loadings)
print(pca.pca.squared_prediction_error)  # use the last column, for the last PCA component

# PLots:
pca.score_plot(pc_horiz=1, pc_vert=2)
pca.loadings_plot(pc_horiz=2, pc_vert=3)
pca.spe_plot(with_a=3)  # for components 1, 2, 3 (inclusive)
pca.T2_plot(with_a=2)
