import pandas as pd
from pytest import approx

from process_improve.multivariate import PCA, MCUVScaler


def test_basic_PCA():
    """
    Arrays with no variance should not be able to have variance extracted.
    """

    foods = pd.read_csv("https://openmv.net/file/food-texture.csv").drop(
        [
            "Unnamed: 0",
        ],
        axis=1,
    )
    scaler = MCUVScaler().fit(foods)
    foods_mcuv = scaler.fit_transform(foods)

    A = 2
    pca = PCA(n_components=A).fit(foods_mcuv)
    T2_limit_95 = pca.T2_limit(0.95)
    assert T2_limit_95 == approx(6.64469, rel=1e-3)

    pca.SPE_limit(0.95)

    ellipse_x, ellipse_y = pca.ellipse_coordinates(1, 2, 0.95, 100)
    assert ellipse_x[-1] == approx(4.48792, rel=1e-5)
    assert ellipse_y[-1] == approx(0, rel=1e-7)
