- **`TPLS.diagnose` reads new F columns by name when F is used as given.** With
  `skip_f_matrix_preprocessing=True`, new F blocks whose columns came in another order than in
  training were scored by position, against the wrong materials' weights, without a word. Every
  block is now taken in the training order first, as centring and scaling already did. (#1028)
