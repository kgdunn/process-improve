- **`interaction_components` and `yates_table` refuse a response that is also named as a factor.**
  `interaction_components(data, "y", ["A", "y"])` split the response by its own three values without
  a word; both now raise `ValueError`: "The response 'y' cannot also be a factor." (#1097)
