#  Copyright (c) 2022 Continental Automotive GmbH
"""Helper functions for setup and evaluation of experiments involving fuzzy logic.

- Experiment preparation: :py:mod:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_helpers`
- Experiment loading and evaluation: :py:mod:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval`
- Visualization: :py:mod:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_vis`

For details on the experiment content see the respective Sacred experiment script.

Experiment results are assumed to be placed within a folder structure
``<experiment_root>/<model_key>/<split>/<formula_dir>/<logic_key>/``
with subfolders ``logs/<run>`` for the sacred logging information
(including the ``config.json`` with the experiment config),
and ``metrics/<timestamp>`` for CSV files containing metric result information.
``split`` will usually be ``TEST``.
"""


from . import fuzzy_exp_helpers as helpers
from . import fuzzy_exp_eval as eval
from . import fuzzy_exp_vis as vis
