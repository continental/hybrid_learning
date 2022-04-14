#  Copyright (c) 2022 Continental Automotive GmbH
"""Sacred experiment for concept analysis on one of several pytorch models
from the model zoo and Broden data.
Specify one of the named configs to start the experiment.
For details on the configuration, have a look at the referenced config scripts.
For an overview, use the sacred ``print_config`` command.
"""

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import

# noinspection PyUnresolvedReferences
from .config import ex  # general settings
# noinspection PyUnresolvedReferences
from .config import config_broden
# noinspection PyUnresolvedReferences
from .config import config_modelzoo  # model zoo settings
# noinspection PyUnresolvedReferences
from .config import config_net2vec  # named config for original Net2Vec settings

if __name__ == "__main__":
    ex.run_commandline()
