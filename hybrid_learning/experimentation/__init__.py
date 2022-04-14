#  Copyright (c) 2022 Continental Automotive GmbH
"""Helper functions for preparing and conducting experiments based on the Sacred framework.
This includes:

- Helpers to register model builders for experiments
  (:py:mod:`~hybrid_learning.experimentation.custom_model_postproc`)
- Helper functions for preparing, conducting, and evaluating experiments

  - on concept embedding analysis
    (:py:mod:`~hybrid_learning.experimentation.ca_exp_eval`) or
  - measuring the truth values of fuzzy logical rules
    (:py:mod:`~hybrid_learning.experimentation.fuzzy_exp`,
    :py:mod:`~hybrid_learning.experimentation.exp_eval_common`)
"""
