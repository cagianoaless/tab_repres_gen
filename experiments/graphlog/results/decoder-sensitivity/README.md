# GraphLog decoder sensitivity

This audit compares fixed logistic/CatBoost, context-tuned CatBoost, a
capacity-restricted depth-1 CatBoost sensitivity, and TabPFN at the registered
40-example GraphLog endpoint.

`catboost_capacity_sensitivity.csv` shows that depth 1 attenuates, but does not
remove, CatBoost's preference for the absolute interface. The result therefore
depends partly on interaction capacity but is not explained by the original
depth-7 configuration alone.
