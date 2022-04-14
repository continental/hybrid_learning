#  Copyright (c) 2022 Continental Automotive GmbH
"""Handles for conducting concept embedding analysis experiments.
Functions and handles to conduct, and log and visualize results
of a concept embedding analysis experiment series.
"""

from .analysis_handle import ConceptAnalysis, EmbeddingReduction, \
    data_for_concept_model
from .results import AnalysisResult, BestEmbeddingResult
