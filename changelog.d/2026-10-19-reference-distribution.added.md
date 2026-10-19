- **`reference_distribution`: judge a difference of averages against the process's own past
  record** (`process_improve.univariate`, #789 B6). From a record in time order and the number of
  observations in each average, it builds every difference of adjacent averages, the later less the
  earlier, and gives the share at least as extreme as an observed difference, a tie counting half
  as Box, Hunter and Hunter count it. It also takes the nearly independent differences (groups of
  2n a gap of one apart) and their root mean square about zero, and gives the t statistic and its
  level on as many degrees of freedom. It reproduces the book's 9 of 191 (0.047) and 0.028 for the
  plant trial, Table 2.5's ten differences and s = 0.60, and the exercises' 0/17, 3/31 and
  16.5/201 = 0.082.
