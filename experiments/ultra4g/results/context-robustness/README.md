# Filtered Negative and Nominal Decoder Control

Effects are paired by dataset after averaging the three registered context seeds. Positive context rows, q64 evaluation queries, candidates, masks, and feature rows are fixed across context modes. Intervals resample the 16 fixed provenance-family effects and are sensitivity summaries rather than superpopulation intervals.

| Contrast | Decoder | Context/interface | Equal-family effect | 95% interval | W/T/L | LOFO | max-|t| p |
|---|---|---|---:|---:|---:|---:|---:|
| local_minus_tokens | catboost_nominal | relation_matched_filtered_uniform | +0.2576 | [+0.1769, +0.3721] | 57/0/0 | [+0.2086, +0.2703] | 0.00198 |
| local_minus_tokens | catboost_nominal | relation_matched_role_degree | +0.1943 | [+0.1183, +0.3139] | 57/0/0 | [+0.1415, +0.2045] | 0.01892 |
| local_minus_tokens | logistic_nominal | relation_matched_filtered_uniform | +0.1971 | [+0.1170, +0.3145] | 57/0/0 | [+0.1455, +0.2074] | 0.01767 |
| local_minus_tokens | logistic_nominal | relation_matched_role_degree | +0.1767 | [+0.0967, +0.3012] | 57/0/0 | [+0.1231, +0.1856] | 0.04459 |
| local_minus_tokens | tabpfn | relation_matched_filtered_uniform | +0.2617 | [+0.1844, +0.3775] | 57/0/0 | [+0.2131, +0.2733] | 0.00143 |
| local_minus_tokens | tabpfn | relation_matched_role_degree | +0.2082 | [+0.1283, +0.3288] | 57/0/0 | [+0.1559, +0.2194] | 0.01291 |
| local_plus_score_minus_score | catboost_nominal | relation_matched_filtered_uniform | +0.0118 | [-0.0539, +0.1245] | 10/2/45 | [-0.0403, +0.0187] | 1.00000 |
| local_plus_score_minus_score | catboost_nominal | relation_matched_role_degree | +0.0635 | [-0.0029, +0.1680] | 39/3/15 | [+0.0168, +0.0752] | 0.95453 |
| local_plus_score_minus_score | logistic_nominal | relation_matched_filtered_uniform | -0.0070 | [-0.0647, +0.0952] | 4/6/47 | [-0.0550, -0.0010] | 1.00000 |
| local_plus_score_minus_score | logistic_nominal | relation_matched_role_degree | -0.0393 | [-0.1152, +0.0762] | 10/3/44 | [-0.0904, -0.0279] | 1.00000 |
| local_plus_score_minus_score | tabpfn | relation_matched_filtered_uniform | +0.0595 | [+0.0053, +0.1579] | 28/12/17 | [+0.0120, +0.0644] | 0.97803 |
| local_plus_score_minus_score | tabpfn | relation_matched_role_degree | +0.0236 | [-0.0385, +0.1281] | 18/6/33 | [-0.0252, +0.0322] | 1.00000 |
| score_given_local | catboost_nominal | relation_matched_filtered_uniform | +0.0481 | [+0.0335, +0.0651] | 48/3/6 | [+0.0424, +0.0513] | 0.00027 |
| score_given_local | catboost_nominal | relation_matched_role_degree | +0.0614 | [+0.0302, +0.0961] | 36/5/16 | [+0.0505, +0.0678] | 0.01773 |
| score_given_local | logistic_nominal | relation_matched_filtered_uniform | +0.1286 | [+0.0928, +0.1622] | 56/0/1 | [+0.1227, +0.1381] | 0.00012 |
| score_given_local | logistic_nominal | relation_matched_role_degree | +0.0833 | [+0.0405, +0.1286] | 41/0/16 | [+0.0703, +0.0916] | 0.01669 |
| score_given_local | tabpfn | relation_matched_filtered_uniform | +0.1243 | [+0.0955, +0.1520] | 54/3/0 | [+0.1179, +0.1326] | 0.00006 |
| score_given_local | tabpfn | relation_matched_role_degree | +0.0558 | [+0.0233, +0.0920] | 38/3/16 | [+0.0439, +0.0640] | 0.05109 |
| structured_minus_filtered_uniform | catboost_nominal | b0_local_only | -0.0633 | [-0.0845, -0.0424] | 4/2/51 | [-0.0674, -0.0598] | 0.00031 |
| structured_minus_filtered_uniform | catboost_nominal | b0_local_ultra | -0.0500 | [-0.0873, -0.0138] | 13/1/43 | [-0.0594, -0.0417] | 0.12988 |
| structured_minus_filtered_uniform | catboost_nominal | entity_tokens_only | -0.0000 | [-0.0009, +0.0009] | 0/56/1 | [-0.0003, +0.0003] | 1.00000 |
| structured_minus_filtered_uniform | catboost_nominal | ultra_score_only | -0.1017 | [-0.1436, -0.0605] | 8/1/48 | [-0.1099, -0.0934] | 0.00284 |
| structured_minus_filtered_uniform | logistic_nominal | b0_local_only | -0.0205 | [-0.0320, -0.0097] | 10/8/39 | [-0.0228, -0.0175] | 0.02060 |
| structured_minus_filtered_uniform | logistic_nominal | b0_local_ultra | -0.0658 | [-0.1016, -0.0321] | 10/4/43 | [-0.0725, -0.0564] | 0.01758 |
| structured_minus_filtered_uniform | logistic_nominal | entity_tokens_only | -0.0001 | [-0.0002, +0.0001] | 0/57/0 | [-0.0001, -0.0000] | 1.00000 |
| structured_minus_filtered_uniform | logistic_nominal | ultra_score_only | -0.0335 | [-0.0731, -0.0048] | 0/45/12 | [-0.0357, -0.0177] | 0.52255 |
| structured_minus_filtered_uniform | tabpfn | b0_local_only | -0.0538 | [-0.0749, -0.0334] | 6/3/48 | [-0.0583, -0.0490] | 0.00168 |
| structured_minus_filtered_uniform | tabpfn | b0_local_ultra | -0.1223 | [-0.1638, -0.0807] | 1/2/54 | [-0.1304, -0.1138] | 0.00049 |
| structured_minus_filtered_uniform | tabpfn | entity_tokens_only | -0.0003 | [-0.0008, +0.0001] | 2/54/1 | [-0.0004, -0.0002] | 0.94077 |
| structured_minus_filtered_uniform | tabpfn | ultra_score_only | -0.0864 | [-0.1323, -0.0435] | 4/13/40 | [-0.0932, -0.0745] | 0.01590 |
