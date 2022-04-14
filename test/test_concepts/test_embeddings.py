"""Tests for concept embeddings."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
import os
from typing import Sequence, Dict, Tuple, List, Optional, Any

import numpy as np
import pytest

from hybrid_learning.concepts.concepts import Concept, SegmentationConcept2D
from hybrid_learning.concepts.models import ConceptDetectionModel2D, ConceptEmbedding
# noinspection PyUnresolvedReferences
from .common_fixtures import concept_model, input_size, sample_layer, \
    concept, main_model  # pylint: disable=unused-import


@pytest.fixture
def concept_embedding(concept_model: ConceptDetectionModel2D
                      ) -> ConceptEmbedding:
    """A concept embedding obtained from a basic concept model."""
    return concept_model.to_embedding()[0]


def to_emb(emb_vals: Sequence[float],
           concept: Concept = None) -> ConceptEmbedding:
    """Given a tuple of (normal_vec, support_factor, scaling_factor) yield
    embedding."""
    assert 2 <= len(emb_vals) <= 3
    concept_args = dict(
        concept=concept, kernel_size=(1,),
        state_dict=dict(
            normal_vec=emb_vals[0],
            bias=-emb_vals[1] * np.linalg.norm(emb_vals[0]) ** 2),
        normal_vec_name="normal_vec", bias_name="bias", )
    if len(emb_vals) == 3:
        concept_args["scaling_factor"] = emb_vals[2]
    return ConceptEmbedding(**concept_args)


def cos_angle(vec1: np.ndarray, vec2: np.ndarray):
    """Math helper: cos(alpha) where alpha is the angle between vectors v, w."""
    return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class TestConceptEmbedding:
    """Test basic functionalities of the ConceptEmbedding class."""

    # Specify the lists to test
    # Entry format: [(emb weight, emb bias), ...], (mean weight, mean bias)
    INTUITIVE_MEAN_EXAMPLES: Dict[str, Tuple[List[Tuple[List[float], float]],
                                             Tuple[List[float], float]]] = \
        {
            "Two identical lines": (
                [([1, 2, 2], 3), ([1.5, 3, 3], 2)],
                ([1 / 3, 2 / 3, 2 / 3], 9)
            ),
            "Two parallels, one not normed, with non-zero bias expected": (
                [([1, 0], 1), ([3, 0], 1)],
                ([1, 0], 2)
            ),
            "Three parallels equally distributed around zero": (
                [([0, 1], 2), ([0, 1], 0), ([0, 1], -2)],
                ([0, 1], 0)
            ),
            "Three parallels, two identical": (
                [([0, 2], 1), ([0, 1], 2), ([0, 1], -2)],
                ([0, 1], 2 / 3)
            ),
            "A cross in zero": (
                [([4, 0, 3], 0), ([4, 0, -3], 0)],
                ([1, 0, 0], 0)
            ),
            "A cross shifted from zero": (
                [([4, 3], 1), ([4, -3], 1)],
                ([1, 0], 6.25)
            ),
            "Two crosses equally distributed around zero": (
                [([4, 3], 1), ([4, -3], 1), ([4, 3], -1), ([4, -3], -1)],
                ([1, 0], 0)
            ),
            "Two crosses shifted from zero": (
                [([4, 3], 3), ([4, -3], 3), ([4, 3], -1), ([4, -3], -1)],
                ([1, 0], 6.25)
            )
        }
    """Tuples of embeddings and mean embedding, each given as
    ``(normal vec, support factor)`` (scaling factors are all 1.). The
    expected mean is the normalized mean of the normal vectors."""

    DISTANCE_MEAN_EXAMPLES: \
        Dict[str, Tuple[List[Tuple[List[float], float]],
                        Tuple[List[float], Optional[float]]]] = \
        {
            "Two identical lines with same scaling": (
                [([1, 2, 2], 2), ([1, 2, 2], 2)],
                ([1, 2, 2], 2)),
            "Two identical lines with different scaling": (
                [([1, 2, 2], 3), ([1.5, 3, 3], 2)],
                ([1.25, 2.5, 2.5], 2.4)),
            "Two parallels, one not normed, with non-zero bias expected": (
                [([1, 0], 1), ([3, 0], 1)],
                ([2, 0], 5 / 4)),
            "Three parallels equally distributed around zero": (
                [([0, 1], 2), ([0, 1], 0), ([0, 1], -2)],
                ([0, 1], 0)),
            "Three parallels equally distributed around zero, one inverted": (
                [([0, 1], 2), ([0, 1], 0), ([0, -1], 2)],
                ([0, 1 / 3], 12)),
            "Three parallels, two identical": (
                [([0, 1], 2), ([0, 1], 2), ([0, -1], 2)],
                ([0, 1 / 3], 18)),
            "Three parallels, two identical but differently scaled": (
                [([0, 2], 1), ([0, 1], 2), ([0, -1], 2)],
                ([0, 2 / 3], 6)),
            "A cross in zero": (
                [([4, 0, 3], 0), ([4, 0, -3], 0)],
                ([4, 0, 0], 0)),
            "A cross shifted from zero": (
                [([4, 3], 1), ([4, -3], 1)],
                ([4, 0], 25 / 16)),
            "Two crosses equally distributed around zero": (
                [([4, 3], 1), ([4, -3], 1), ([4, 3], -1), ([4, -3], -1)],
                ([4, 0], 0)),
            "Two crosses shifted from zero": (
                [([4, 3], 3), ([4, -3], 3), ([4, 3], -1), ([4, -3], -1)],
                ([4, 0], 25 / 16)),
            "Cross with differently scaled normal vectors": (
                [([2, 1], 2), ([4, -2], 1)],
                ([3, -0.5], 15 / 9.25))
        }
    """Examples for calculating the mean embedding by distance.
    Format: {description: ([input embeddings], output embedding} and an
    embedding is given by ``(normal vector, support_factor)``, with support
    factor possibly not given (i.e. None)."""

    TEST_POINTS_nD = {
        2: ([0, 0], [1, 1], [-1, 1], [1, -1], [-1, -1], [3, 4], [4, 3]),
        3: ([0, 0, 0], [1, 1, 1], [-1, 1, 1], [1, -1, -1], [-1, -1, -1],
            [3, 4, 0], [4, 0, 3])
    }
    """Testing points to check mean conditions for several dimensions."""

    DISTANCE_EXAMPLES = [
        (((1, 0), 3, 1), (0, 0), -3),
        (((1, 0), 3, -1), (0, 0), 3),
        (((0, 1), 3, 1), (0, 0), -3),
        (((1 / 3, 2 / 3, 2 / 3), 9, 1), (0, 0, 0), -9)
    ]
    """Examples for calculating the distance of a point to an embedding
    hyperplane.
    Format: Tuples of
    - embedding as (normal_vec, support_factor, scaling_factor)
    - point
    - distance
    """

    def test_str(self, concept_embedding: ConceptEmbedding):
        """Test the string function."""
        # printing should not rise error
        str(concept_embedding)

    def test_save_and_load(self, concept_embedding: ConceptEmbedding,
                           tmp_path: str):
        """Test save and load."""
        filepath: str = os.path.join(tmp_path, "test.pt")

        concept_embedding.save(filepath)
        emb = ConceptEmbedding.load(filepath)

        # torch.nn.Module and Concept aren't saved:
        assert concept_embedding.main_model is not None
        assert concept_embedding.concept is not None
        assert emb.main_model is None
        assert emb.concept is None

        # Rest is the same:
        assert emb == ConceptEmbedding(**{
            k: v for k, v in concept_embedding.settings.items()
            if k not in ('main_model', 'concept')})

    def test_legacy_load(self, concept_embedding: ConceptEmbedding,
                         tmp_path: str):
        """Test the legacy loading from .npz files."""
        filepath: str = os.path.join(tmp_path, "test.npz")

        # legacy saved embedding:
        setts: Dict[str, Any] = dict(
            normal_vec=concept_embedding.normal_vec,
            support_factor=concept_embedding.support_factor,
            scaling_factor=concept_embedding.scaling_factor,
            concept_name=str(concept_embedding.concept_name),
            layer_id=str(concept_embedding.layer_id))
        with open(filepath, 'w+b') as npz_file:
            np.savez(npz_file, **setts)

        emb: ConceptEmbedding = ConceptEmbedding.load(filepath)
        assert emb == concept_embedding

        # Proper defaults for concept and main model:
        assert emb.concept is None
        assert emb.main_model is None

    def test_normalize(self):
        """Test fundamental properties of normalization."""
        normal_vec = np.array([3, 4])  # |normal_vec| = 5
        support_factor = 1
        scaling_factor = 1.
        normed_normal_vec = np.array([3 / 5, 4 / 5])
        normed_support_factor = 5
        normed_scaling_factor = 5
        # noinspection PyTypeChecker
        emb = ConceptEmbedding(
            state_dict=dict(
                normal_vec=normal_vec,
                bias=-support_factor * np.linalg.norm(normal_vec) ** 2),
            normal_vec_name="normal_vec", bias_name="bias", kernel_size=(1,))
        assert emb.scaling_factor == scaling_factor, \
            "Scaling factor wrongly initialized."
        normed_emb = emb.normalize()

        # Normalization yields new instance
        assert normed_emb is not emb, "Normalization did not yield new instance"

        # Format checks
        assert normed_emb.normal_vec.shape == emb.normal_vec.shape
        assert np.array(normed_emb.support_factor).shape == np.array(
            emb.support_factor).shape
        assert np.array(normed_emb.scaling_factor).shape == np.array(
            emb.scaling_factor).shape

        # Value checks
        for key, (expected, obtained) in \
                {"normal_vec": (normed_normal_vec, normed_emb.normal_vec),
                 "support_factor": (normed_support_factor,
                                    normed_emb.support_factor),
                 "scaling_factor": (normed_scaling_factor,
                                    normed_emb.scaling_factor)
                 }.items():
            assert np.allclose(obtained, expected), \
                ("Wrong normalized {}: expected {}, but was {}"
                 .format(key, expected, obtained))

    def test_scale(self):
        """Test fundamental properties of applying the scaling factor."""
        # normal_vec, support_factor, scaling_factor
        orig = ([3, 4], 1, 1)
        normal_vec, support_factor, scaling_factor = orig
        # noinspection PyTypeChecker
        emb = ConceptEmbedding(
            state_dict=dict(
                normal_vec=normal_vec,
                bias=-support_factor * np.linalg.norm(normal_vec) ** 2),
            normal_vec_name="normal_vec", bias_name="bias", kernel_size=(1,),
            scaling_factor=scaling_factor)

        # Scaling yields new instance and old one is not changed:
        assert emb is not emb.scale(), "Scaling did not yield new instance"
        assert np.allclose(emb.normal_vec, np.array(normal_vec))
        assert np.allclose(emb.support_factor, np.array(support_factor))
        assert np.allclose(emb.scaling_factor, np.array(scaling_factor))

        # Scaling embedding of scaling factor 1 does nothing
        scaled_unchanged_emb = emb.scale()
        assert np.allclose(emb.normal_vec, scaled_unchanged_emb.normal_vec)
        assert np.allclose(emb.support_factor,
                           scaled_unchanged_emb.support_factor)
        assert np.allclose(emb.scaling_factor,
                           scaled_unchanged_emb.scaling_factor)

        # Normalization and then scaling should yield the same embedding:
        backscaled_emb = emb.normalize().scale()
        for key, (expected, obtained) in \
                {"normal_vec": (emb.normal_vec, backscaled_emb.normal_vec),
                 "support_factor": (emb.support_factor,
                                    backscaled_emb.support_factor),
                 "scaling_factor": (emb.scaling_factor,
                                    backscaled_emb.scaling_factor)
                 }.items():
            assert np.allclose(obtained, expected), \
                ("Wrong normalized {}: expected {}, but was {}"
                 .format(key, expected, obtained))

        # Simple scaling example: scale by 2
        emb.scaling_factor = 2
        new = ([6, 8], 0.5, 1)
        normal_vec, support_factor, scaling_factor = new
        scaled_emb = emb.scale()
        assert np.allclose(normal_vec, scaled_emb.normal_vec)
        assert np.allclose(support_factor, scaled_emb.support_factor)
        assert np.allclose(scaling_factor, scaled_emb.scaling_factor)

        # Another scaling example: scale by -2
        emb.scaling_factor = -2
        new = ([-6, -8], -0.5, 1)
        normal_vec, support_factor, scaling_factor = new
        scaled_emb = emb.scale()
        assert np.allclose(normal_vec, scaled_emb.normal_vec)
        assert np.allclose(support_factor, scaled_emb.support_factor)
        assert np.allclose(scaling_factor, scaled_emb.scaling_factor)

    def test_upper_sphere_neg_bias(self):
        """Test embedding normalization with negative bias."""
        weight = np.array([-3, 4])  # |weight| = 5
        support_factor = -1
        normed_weight = np.array([3 / 5, -4 / 5])
        normed_support_factor = 5
        # noinspection PyTypeChecker
        emb = ConceptEmbedding(
            state_dict=dict(normal_vec=weight,
                            bias=-support_factor * np.linalg.norm(weight) ** 2),
            normal_vec_name="normal_vec", bias_name="bias", kernel_size=(1,))
        normed_emb = emb.unique_upper_sphere()

        # Format checks
        assert normed_emb.normal_vec.shape == emb.normal_vec.shape
        assert np.array(normed_emb.support_factor).shape == np.array(
            emb.support_factor).shape

        # Value checks
        assert np.allclose(normed_emb.normal_vec, normed_weight), \
            ("Wrong normalized weight: expected {}, but was {}"
             .format(normed_weight, normed_emb.normal_vec))
        assert np.allclose(normed_emb.support_factor, normed_support_factor), \
            ("Wrong normalized bias: expected {}, but was {}"
             .format(normed_support_factor, normed_emb.support_factor))

    def test_distance(self):
        """Test some fundamental properties of the distance function for
        embeddings."""
        for emb_vals, point, dist_gt in self.DISTANCE_EXAMPLES:
            print(emb_vals, point, dist_gt)
            emb = to_emb(emb_vals)
            dist = emb.distance(point)
            assert np.allclose(dist, dist_gt), \
                ("Wrong distance for point {}: expected {} but was {};"
                 "\nembedding:\n{}").format(point, dist_gt, dist, str(emb))

    def test_distance_mean_exceptions(self, concept: SegmentationConcept2D):
        """Test obtaining the mean concept embedding, meaning the optimal mean
        hyperplane. Are appropriate errors raised?"""
        with pytest.raises(ValueError):
            # Two parallels equally distributed around zero but with opposite
            # pos/neg sides
            concept_args = dict(concept=concept)
            embeddings = [
                ConceptEmbedding(
                    state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                    normal_vec_name="w", bias_name="B",
                    kernel_size=(1,), **concept_args)
                for w, b in [([0, 1], 2), ([0, -1], 2)]]
            ConceptEmbedding.mean_by_distance(embeddings)

        with pytest.raises(ValueError):
            ConceptEmbedding.mean_by_distance([])

    def test_distance_mean_results(self, concept: Concept):
        """Test obtaining the mean concept embedding by sample values."""

        concept_args = dict(concept=concept)
        for desc, (embs, (m_w, m_b)) in self.DISTANCE_MEAN_EXAMPLES.items():
            m_w: np.ndarray = np.array(m_w)
            embeddings = [
                ConceptEmbedding(
                    state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                    normal_vec_name="w", bias_name="B",
                    kernel_size=(1,), **concept_args)
                for w, b in embs]
            # Actual routine
            m_emb: ConceptEmbedding = \
                ConceptEmbedding.mean_by_distance(embeddings)
            context_info = (("context:\n  mean embedding: ({}, {}, 1.)"
                             "\n  in embeddings ({}) as (normal vec, support, "
                             "scaling):\n   {}")
                            .format(m_emb.normal_vec, m_emb.support_factor,
                                    desc,
                                    [(e.normal_vec, e.support_factor,
                                      e.scaling_factor) for e in embeddings]))

            # Format checks
            assert m_emb.normal_vec.shape == embeddings[0].normal_vec.shape
            assert np.array(m_emb.support_factor).shape == np.array(
                embeddings[0].support_factor).shape

            # Value checks
            # the embedding should be scaled
            assert float(m_emb.scaling_factor) == 1., \
                ("Mean embedding not scaled: expected 1., but was {}; {}"
                 .format(m_emb.scaling_factor, context_info))
            assert np.allclose(m_emb.normal_vec, m_w), \
                ("Wrong mean normal vector: expected {}, but was {}; {}"
                 .format(m_w, m_emb.normal_vec, context_info))
            # For all given ground truths of support factors, check them:
            if m_b is not None:
                assert np.allclose(m_emb.support_factor, m_b), \
                    ("Wrong mean support factor: expected {}, but was {}; {}"
                     .format(m_b, m_emb.support_factor, context_info))

    def test_distance_mean_condition(self, concept: Concept):
        """Test that the mean embedding by distance really fulfills the
        distance condition.
        This is that at any point in space the distance of the mean embedding is
        the mean of the distances of the other embeddings."""
        # test points for testing the distance condition in several dimensions
        concept_args = dict(concept=concept)
        for desc, (embs, _) in self.DISTANCE_MEAN_EXAMPLES.items():
            embeddings = [
                ConceptEmbedding(
                    state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                    normal_vec_name="w", bias_name="B",
                    kernel_size=(1,), **concept_args)
                for w, b in embs]
            # Actual routine
            m_emb: ConceptEmbedding = \
                ConceptEmbedding.mean_by_distance(embeddings)
            dims: int = m_emb.normal_vec.size
            test_points = self.TEST_POINTS_nD[dims]
            for point in test_points:
                dist_mean: float = m_emb.distance(point)
                dists: List[float] = [e.distance(point) for e in embeddings]
                mean_dist: float = float(np.mean(dists))
                assert np.allclose(dist_mean, mean_dist), \
                    (("In point {},  distance of mean ({}) is not mean of "
                      "distances {} ({}) for\n  in embeddings: {}:\n   {}"
                      "\n  mean embedding: {}")
                     .format(point, dist_mean, dists, mean_dist,
                             desc, [(e.normal_vec, e.support_factor)
                                    for e in embeddings],
                             (m_emb.normal_vec, m_emb.support_factor)))

    def test_mean_results(self, concept: Concept):
        """Test obtaining the mean concept embedding by sample values."""
        concept_args = dict(concept=concept)
        for desc, (embs, (m_w, m_b)) in self.INTUITIVE_MEAN_EXAMPLES.items():
            m_w: np.ndarray = np.array(m_w)
            embeddings = [
                ConceptEmbedding(
                    state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                    normal_vec_name="w", bias_name="B",
                    kernel_size=(1,), **concept_args)
                for w, b in embs]
            # Actual routine
            m_emb: ConceptEmbedding = ConceptEmbedding.mean(embeddings)
            context_info = (("context:\n  mean embedding: ({}, {}, 1.)"
                             "\n  in embeddings ({}) as (normal vec, support, "
                             "scaling):\n   {}")
                            .format(m_emb.normal_vec, m_emb.support_factor,
                                    desc,
                                    [(e.normal_vec, e.support_factor,
                                      e.scaling_factor)
                                     for e in embeddings]))

            # Format checks
            assert m_emb.normal_vec.shape == embeddings[0].normal_vec.shape
            assert np.array(m_emb.support_factor).shape == np.array(
                embeddings[0].support_factor).shape

            # Value checks
            # Scaling
            expected_scaling: float = float(np.mean(
                [e.scaling_factor for e in
                 [e.normalize() for e in embeddings]]))
            assert float(m_emb.scaling_factor) == expected_scaling, \
                ("Mean scaling wrong: expected {}., but was {}; {}"
                 .format(expected_scaling, m_emb.scaling_factor, context_info))
            # Normal vector
            assert np.allclose(m_emb.normal_vec, m_w), \
                ("Wrong mean normal vector: expected {}, but was {}; {}"
                 .format(m_w, m_emb.normal_vec, context_info))
            # Support
            assert np.allclose(m_emb.support_factor, m_b), \
                ("Wrong mean support factor: expected {}, but was {}; {}"
                 .format(m_b, m_emb.support_factor, context_info))

    def test_complex_mean_exceptions(self):
        """Test obtaining the mean concept embedding, meaning the optimal
        mean hyperplane. Are appropriate errors raised?"""
        with pytest.raises(ValueError):
            ConceptEmbedding.mean_by_angle([])

    def test_complex_mean(self, concept: Concept):
        """Test obtaining the mean concept embedding, meaning the optimal
        mean hyperplane."""

        concept_args = dict(concept=concept)
        for desc, (embs, (m_w, m_b)) in self.INTUITIVE_MEAN_EXAMPLES.items():
            m_w: np.ndarray = np.array(m_w)
            embeddings = [ConceptEmbedding(
                state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                normal_vec_name="w", bias_name="B",
                kernel_size=(1,), **concept_args)
                for w, b in embs]
            # Actual routine
            m_emb: ConceptEmbedding = ConceptEmbedding.mean_by_angle(embeddings)
            context_info = (("context:\n  mean embedding: ({}, {})"
                             "\n  in embeddings ({}) as (normal vec, support, "
                             "scaling):\n   {}")
                            .format(m_emb.normal_vec, m_emb.support_factor,
                                    desc,
                                    [(e.normal_vec, e.support_factor,
                                      e.scaling_factor) for e in embeddings]))

            # Format checks
            assert m_emb.normal_vec.shape == embeddings[0].normal_vec.shape
            assert np.array(m_emb.support_factor).shape == np.array(
                embeddings[0].support_factor).shape

            # Value checks
            assert np.allclose(m_emb.normal_vec, m_w), \
                ("Wrong mean normal weight: expected {}, but was {}; {}"
                 .format(m_w, m_emb.normal_vec, context_info))
            assert np.allclose(m_emb.support_factor, m_b), \
                ("Wrong mean normal bias: expected {}, but was {}; {}"
                 .format(m_b, m_emb.support_factor, context_info))

    def test_complex_mean_condition(self, concept: Concept):
        """Test obtaining the mean concept embedding, meaning the optimal mean
        hyperplane.
        The condition for the more complex mean here is that at any point the
        mean distances to the hyperplanes along the normal vector equal the
        (unscaled!) distance to the mean hyperplane."""
        concept_args = dict(concept=concept)
        for desc, (embs, _) in self.INTUITIVE_MEAN_EXAMPLES.items():
            embs: List[ConceptEmbedding] = [
                ConceptEmbedding(
                    state_dict=dict(w=w, B=-b * np.linalg.norm(w) ** 2),
                    normal_vec_name="w", bias_name="B",
                    kernel_size=(1,), **concept_args)
                for w, b in embs]
            # Actual routine
            m_emb: ConceptEmbedding = ConceptEmbedding.mean_by_angle(embs) \
                .normalize().to_pos_scaling()
            context_info = (
                ("embeddings in context as (normal vec, support, scaling):"
                 "\n  mean embedding: ({}, {}, {})"
                 "\n  in embeddings ({}):\n   {}"
                 ).format(m_emb.normal_vec, m_emb.support_factor,
                          m_emb.scaling_factor,
                          desc,
                          [(e.normal_vec, e.support_factor, e.scaling_factor)
                           for e in embs]))

            for point in self.TEST_POINTS_nD[m_emb.normal_vec.size]:
                # The normalized vectors with no scaling for distance
                # calculation
                # pylint: disable=no-member
                normed_embs: List[ConceptEmbedding] = \
                    [e.normalize().to_pos_scaling().forget_scaling()
                     for e in embs]
                # pylint: enable=no-member
                # The for each hyperplane the cosine of its angle to the mean
                # hyperplane
                cos_angles = [cos_angle(e.normal_vec, m_emb.normal_vec)
                              for e in normed_embs]
                # Distance from each hyperplane to the point along the
                # hyperplane's normal vec
                plane_to_pt_dists: List[float] = [e.distance(point) for e in
                                                  normed_embs]
                # Distances from the point to each hyperplane along the mean
                # normal vec
                dists: List[float] = [d / cos_a for d, cos_a in
                                      zip(plane_to_pt_dists, cos_angles)]
                # Condition: The distances along the mean normal vec should
                # sum up to the mean dist
                dist_to_mean = m_emb.forget_scaling().distance(point)
                assert np.allclose(dist_to_mean, np.mean(dists)), \
                    (("In point {}, distance to mean hyperplane is not the "
                      "mean distance to the other hyperplanes along the mean "
                      "hyperplane normal vector!"
                      "\ndists to hyperplanes along mean normal vec: {}, "
                      "mean: {};\nunscaled dist to mean hyperplane: {}; {}")
                     .format(point, dists, np.mean(dists), dist_to_mean,
                             context_info))

    def test_equals(self):
        """Test equality between two concept embeddings."""
        args = dict(concept=None)

        def to_emb(normal_vec, support_factor, scaling_factor):
            """Shorthand to concept embedding creation"""
            return ConceptEmbedding(
                state_dict=dict(
                    normal_vec=normal_vec,
                    bias=-support_factor * np.linalg.norm(normal_vec) ** 2),
                normal_vec_name="normal_vec", bias_name="bias",
                kernel_size=(1,), scaling_factor=scaling_factor, **args)

        # self == self
        emb = to_emb([1, 3, 3, 2], 45, 3)
        # pylint: disable=comparison-with-itself
        assert emb == emb
        # pylint: enable=comparison-with-itself
        assert emb == to_emb([1, 3, 3, 2], 45, 3)

        # Some positive example values:
        assert to_emb([1, 0], 0, 3) == to_emb([3, 0], 0, 1)
        assert to_emb([0, 4], 1, 1) == to_emb([0, 1], 4, 4)

        # Some negative example values:
        # Different normal vector
        assert to_emb([1, 0], 0, 1) != to_emb([0, 1], 0, 1)
        # Different support
        assert to_emb([0, 4], -1, 1) != to_emb([0, 1], 4, 4), \
            "Normalized versions equal:\n{}\n{}".format(
                to_emb([0, 4], -1, 1).normalize(),
                to_emb([0, 1], 4, 4).normalize())
        # Different scaling factor (the same hyperplane but with
        # anti-parallel normal vector)
        assert to_emb([5, 4], 1, 1) != to_emb([5, 4], -1, -1)
        assert to_emb([1, 0], 0, -3) != to_emb([3, 0], 0, 1)
