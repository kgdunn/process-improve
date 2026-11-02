- **`PRM` keeps a row out of the fit when its weight underflows to 0.** A row far out in both X
  and y (1e100 standardised units, say) has a residual weight and a leverage weight whose product
  underflows to 0. That 0 was replaced with 1e-6, as the R package chemometrics does, and at that
  weight the row's size let it lead the next weighted fit: the loop ran to `max_weight_iter`
  without converging, and the predictions of the other rows moved by three times y's standard
  deviation. The weight now stays 0, so `robust_weights_` lies in [0, 1], and the fit is the one
  the other rows give. Data without such a row give the same model as before. (#1087)
