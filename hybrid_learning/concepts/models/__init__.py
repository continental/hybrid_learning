"""Models and handles to define and store concept embeddings.
This encompasses

- model architecture definitions
- an implementation independent format for representing a concept embedding
  obtained from a linear model, i.e. with a concept vector (the weights)
  and an offset (the bias)
- wrappers to extend and (re-)attach to DNNs
"""

#  Copyright (c) 2022 Continental Automotive GmbH

from . import embeddings
from .concept_models import *
from .embeddings import ConceptEmbedding
