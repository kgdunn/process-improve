# --------------


# -------


tpls_example = load_tpls_example()
estimator = TPLSpreprocess()
estimator.fit(tpls_example)
transformed_data = estimator.transform(tpls_example)


# class Plot:
# def __init__(self):
#     pass


class Plot:
    """Make plots of estimators."""

    # _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    # _series_kinds = ("pie",)
    # _dataframe_kinds = ("scatter", "hexbin")
    # _kind_aliases = {"density": "kde"}
    # _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, parent: BaseEstimator) -> None:
        self._parent = parent

    # def __call__(self, *args, **kwargs):
    #    plot_backend = do_stuff(kwargs.pop("backend", None))

    def scores(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:  # noqa: ARG002
        """Generate a scores plot."""
        print(f"generate scores plot with {pc_horiz} horizontal and {pc_vert}")  # noqa: T201

        return go.Figure()


estimator = TPLSpreprocess()
transformed_data = estimator.fit_transform(load_tpls_example())

fitted = TPLS(n_components=3)
fitted.fit(transformed_data)
plot_settings = dict(plotwidth=1000, title="Z Space")
fitted.plot.scores(pc_horiz=1, pc_vert=2, **plot_settings)
# fitted.plot.loadings("Z")
# fitted.plot.loadings("F")
# fitted.plot.loadings("D")
# fitted.plot.loadings("D", "Y")
# fitted.plot.vip()
# fitted.plot.r2()


# Predict a new blend
# rnew = {
#     "MAT1": [("A0129", 0.557949425), ("A0130", 0.442050575)],
#     "MAT2": [("Lac0003", 1)],
#     "MAT3": [("TLC018", 1)],
#     "MAT4": [("M0012", 1)],
#     "MAT5": [("CS0017", 1)],
# }
# znew = process[process["LotID"] == "L001"]
# znew = znew.values.reshape(-1)[1:].astype(float)
# # preds = phi.tpls_pred(rnew, znew, tplsobj)
# fitted.predict(transformed_data)

test_tpls_model_fitting()
