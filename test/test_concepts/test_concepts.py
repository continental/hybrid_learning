"""Tests for functionality of concepts."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=redefined-outer-name
from hybrid_learning.concepts.concepts import Concept
# noinspection PyUnresolvedReferences
from .common_fixtures import concept, \
    input_size  # pylint: disable=unused-import


def test_data_info(concept: Concept):
    """Simply run the info function to see whether it produces an error."""
    print()
    print(concept.data.info)
