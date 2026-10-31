- **`AdaptivePCA` and `AdaptivePLS` keep adapting a tag after a missing value.** An update with a
  missing value made that tag's moving centre and scale NaN, for good: the tag was then treated
  as missing in every later observation and dropped out of the model. A missing value now moves
  neither its tag's centre nor its scale, and the other tags update as before. (#1029)
