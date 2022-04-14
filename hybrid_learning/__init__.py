#  Copyright (c) 2022 Continental Automotive GmbH

"""Library for concept embedding analysis, model extension, and fuzzy logic rule formulation.
This pytorch-based library provides functions and handles to associate visual semantic concepts
in DNN latent spaces with so-called concept embeddings, which are linear models
that predict information about the concept from latent space information.

Core features:

- Models, training and experiment handles for finding concept embeddings
  (:py:mod:`hybrid_learning.concepts`)
- Wrappers to allow dissecting and (re-)attaching pytorch DNN models and model parts
  (:py:mod:`hybrid_learning.concepts.models.model_extension`)
- Collection of convenient dataset wrappers, transformations, and caching mechanisms
  (:py:mod:`hybrid_learning.datasets`)
- A DSL-like library for defining and parsing formal fuzzy logic rules to transformation functions
  (:py:mod:`fuzzy_logic`)
"""
