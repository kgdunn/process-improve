import numpy as np


def nan_to_zeros(in_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NaN to zero and return a NaN map."""

    nan_map = np.isnan(in_array)
    in_array[nan_map] = 0
    return in_array, nan_map


def regress_y_space_on_x(y_space: np.ndarray, x_space: np.ndarray, y_space_nan_map: np.ndarray) -> np.ndarray:
    """
    Project the rows of `y_space` onto the vector `x_space`. Neither of these two inputs may have missing values.

    The `y_space_nan_map` has `True` entries where `y_space` originally had NaN values. The `x_space` may never have
    missing values.

    y_space = [n_rows x j_cols]
    x_space = [j_cols x 1]
    Returns   [n_rows x 1]
    """

    b_mat = np.tile(x_space.T, (y_space.shape[0], 1))  # tiles, row-by-row the `x_space` row vector, to create `n_rows`
    denominator = np.sum((b_mat * ~y_space_nan_map) ** 2, axis=1).astype("float")
    denominator[denominator == 0] = np.nan
    return np.array((np.sum(y_space * b_mat, axis=1)) / denominator).reshape(-1, 1)


def test_nan_to_zeros() -> None:
    """Test the `nan_to_zeros` function."""
    in_array = np.array([[1, 2, np.nan], [4, 5, 6], [float("nan"), 8, 9]])
    out_array, nan_map = nan_to_zeros(in_array)
    assert np.allclose(out_array, np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]))
    assert np.allclose(nan_map, np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]))


def test_regress_y_space_on_x() -> None:
    """Test the `regress_y_space_on_x` function."""
    x_space = np.array([1, 2, 3, 4])
    y_space = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, float("NaN"), 4],
            [float("NaN"), float("NaN"), 3, float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), 4],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [6, 4, 2, 0],
        ]
    )

    y_space_filled, y_space_nan_map = nan_to_zeros(y_space)
    regression_vector = regress_y_space_on_x(y_space_filled, x_space, y_space_nan_map)
    assert np.allclose(regression_vector, np.array([[1, 1, 1, 1, float("nan"), 2 / 3]]).T, equal_nan=True)


test_nan_to_zeros()
test_regress_y_space_on_x()
