"""Representation of concept embeddings and modification functions."""
#  Copyright (c) 2020 Continental Automotive GmbH

from typing import Union, Sequence, List, Tuple, Optional, Any, Dict, \
    TYPE_CHECKING

import numpy as np
import torch.nn

from .concepts import Concept

if TYPE_CHECKING:
    # avoid cyclic import for the typing stuff
    from .models.model_extension import ModelStump


# noinspection PyArgumentEqualDefault
class ConceptEmbedding:
    r"""Representation of an embedding of a concept within a DNN.
    The representation aims to be technology independent.

    Main aspects:

    - the parameters:
      :math:`\text{concept vector} = \text{weight}`,
      :math:`\text{bias} = -\text{threshold}`
    - the layer it is attached to given by the model up to that layer.
    """

    @property
    def concept_name(self) -> Optional[str]:
        """Shortcut to the name of the concept that is embedded."""
        return self._concept_name if self.concept is None else self.concept.name

    @property
    def layer_id(self) -> str:
        """The layer in the main model in which the concept is embedded."""
        return self.model_stump.stump_head \
            if self.model_stump is not None else self._layer_id

    @property
    def main_model(self) -> torch.nn.Module:
        """The main torch model in which the concept is embedded."""
        return self.model_stump.wrapped_model \
            if self.model_stump is not None else None

    def scale(self) -> 'ConceptEmbedding':
        """Return a new equivalent embedding with ``scaling_factor == 1``."""
        scaled_normal_vec = self.normal_vec * self.scaling_factor
        scaled_support_factor = self.support_factor / self.scaling_factor
        return ConceptEmbedding(normal_vec=scaled_normal_vec,
                                support_factor=scaled_support_factor,
                                scaling_factor=1.,
                                **self.meta_info)

    def save(self, filepath: str, overwrite: bool = True):
        """Save the embedding parameters and some description as npz file.
        The file contains values for the following keys:
        ``normal_vec``, ``scaling_factor``, ``support_factor``,
        ``concept_name``, ``layer_id``.
        Load the embedding using :py:meth:`load`.
        """
        mode = 'w+b' if overwrite else 'xb'
        settings = dict(normal_vec=self.normal_vec,
                        support_factor=self.support_factor,
                        scaling_factor=self.scaling_factor,
                        concept_name=str(self.concept_name),
                        layer_id=str(self.layer_id))
        with open(filepath, mode) as npz_file:
            np.savez(npz_file, **settings)

    @classmethod
    def load(cls, filepath: str):
        """Load an embedding.
        The format should be as used by :py:meth:`save`."""
        return cls(**np.load(filepath, allow_pickle=False))

    @property
    def meta_info(self) -> Dict[str, Any]:
        """Meta information about the embedding (model and concept)."""
        return dict(concept=self.concept,
                    model_stump=self.model_stump,
                    concept_name=self.concept_name,
                    layer_id=self.layer_id)

    @property
    def settings(self) -> Dict[str, Any]:
        """Dictionary to reproduce the instance."""
        return dict(normal_vec=self.normal_vec,
                    support_factor=self.support_factor,
                    scaling_factor=self.scaling_factor,
                    **self.meta_info)

    def __init__(self, normal_vec: np.ndarray,
                 support_factor: Union[np.ndarray, float],
                 scaling_factor: Union[np.ndarray, float] = 1.,
                 concept: Concept = None,
                 model_stump: 'ModelStump' = None,
                 concept_name: str = None, layer_id: str = None):
        """Init.

        :param concept: the concept that is embedded
        :param model_stump: the model up to the layer of the embedding
        :param normal_vec: the concept vector
        :param support_factor: the negative concept threshold
        :param layer_id: if ``model_stump`` is not given, optional
            specification of the ``layer_id``
        :param concept_name: if ``concept`` is not given,
            optional specification of the name
        """
        self.concept: Concept = concept
        """The concept that is embedded."""
        self._concept_name: Optional[str] = concept_name \
            if self.concept is None else self.concept.name
        """If the concept is not given, identifier for the concept used in
        :py:attr`concept_name`"""

        # where the concept model is attached
        self.model_stump: 'ModelStump' = model_stump
        """The head of the model and the layer the concept is embedded into."""
        self._layer_id = str(layer_id) if model_stump is None else None
        """If the model stump is not given, the layer ID used in
        :py:attr:`layer_id`"""

        # parameters of the concept model
        self.normal_vec: np.ndarray = np.array(normal_vec)
        """A normal vector to the represented hyperplane."""
        self.support_factor: np.ndarray = np.array(support_factor).reshape(-1)
        r"""A factor :math:`b` to obtain the orthogonal support vector
        :math:`b\cdot n` from the normal vector :math:`n`.
        A vector :math:`v` is on the hyperplane iff

        .. math:: 0 = d(v) = (v - b\cdot n) \circ n = v \circ n - b\cdot |n|^2

        Here, :math:`d(v)` denotes the signed orthogonal distance of :math:`v`
        from the hyperplane.
        """
        if not self.support_factor.size == 1:
            raise ValueError(
                ("Got bias which has more than one entry (shape: {}): {}"
                 ).format(np.array(support_factor).shape, support_factor))

        self.scaling_factor: np.ndarray = scaling_factor
        r"""The factor to obtain the original weight vector.
        Any two embeddings with normal vectors :math:`n_1, n_2` and support
        factors :math:`b_1, b_2` fulfilling the following represent the same
        hyperplane:

        .. math::
            :nowrap:

            \begin{align*}
            \frac{|n_1 \circ n_2|} {(|n_1| \cdot |n_2|)} &= 1 &\text{and} &&
            \frac{|n_1|} {|n_2|} &= \frac{b_2} {b_1}
            \end{align*}

        However, the signed orthogonal distance measure of an embedding
        :math:`(n, b)` for a vector :math:`v`

        .. math::
            d(v)
            = (v - b \cdot n) \circ n
            = |n| \cdot \left(v \circ \frac{n}{|n|}\right) - b\cdot|n|^2

        which is used e.g. in concept layers, depends quadratic on the normal
        vector length. If the hyperplane representation is changed,
        the original normal vector and support factor
        providing the original distance measure can be obtained via

        .. math::
            \left(n \cdot \text{scaling_factor} ,
            \frac{b}{\text{scaling_factor}})\right.

        *Examples:*
        The scaling_factor is 1 if the original weight was not changed,
        and :math:`|weight|` if it was normalized.
        """

    def distance(self, point: np.ndarray) -> float:
        r"""Calc the scaled distance of point from the embedding hyperplane.
        The distance from a point pt is given by

        .. math::
            d(pt) =
            \text{scaling_factor} \cdot \left((n \circ pt)
            - b (n \circ n)\right)
        """
        point = np.array(point)
        if point.shape != self.normal_vec.shape:
            raise ValueError(("Cannot calculate distance for point of different"
                              " shape than normal vec; expected {} but was {}"
                              ).format(self.normal_vec.shape, point.shape))

        # d(pt) = scaling_factor * [(n \circ pt) - b * (n \circ n)]
        dist: float = (self.scaling_factor *
                       (np.sum(point * self.normal_vec)
                        - self.support_factor
                        * np.sum(self.normal_vec * self.normal_vec)))
        return float(dist)

    def normalize(self) -> 'ConceptEmbedding':
        """Yield a new, equivalent embedding with normalized normal vec.
        The sign of the scaling factor is not changed."""
        normal_vec_norm = np.linalg.norm(self.normal_vec)
        if np.allclose(normal_vec_norm, 0):
            raise ValueError("Tried to normalize zero vector.")
        normed_normal_vec = self.normal_vec / normal_vec_norm
        normed_scaling_factor = normal_vec_norm * self.scaling_factor
        normed_support_factor = normal_vec_norm * self.support_factor
        return ConceptEmbedding(normal_vec=normed_normal_vec,
                                support_factor=normed_support_factor,
                                scaling_factor=normed_scaling_factor,
                                **self.meta_info)

    def unique(self) -> 'ConceptEmbedding':
        """Yield new, equivalent, unique embedding with normalized normal vec
        and pos scaling."""
        return self.normalize().to_pos_scaling()

    def unique_upper_sphere(self) -> 'ConceptEmbedding':
        r"""Yield new equivalent, unique embedding with normal vec normalized
        in upper hemisphere.

        An embedding defines a hyperplane as follows:

        - the :math:`weight` is a (not necessarily normalized) normal vector
          of the hyperplane
        - :math:`bias \cdot weight` is a support vector orthogonal to the plane

        This representation is not unique.
        In many cases it is desirable to consider the representation where
        the normal vector is normalized, and lies on the upper half of a
        given sphere (including the equator). To also obtain unique results
        for the equator cases, the rule is that, when flattened, the first
        non-zero entry is positive.
        The representation obtained then as follows is unique
        (sign(weight) is the sign of the first non-zero entry when flattened):

        .. math::
            weight_{new} &= sign(weight) \cdot \frac{weight} {|weight|} \\
            bias_{new}   &= sign(weight) \cdot (bias \cdot |weight|)

        Then the weight is normalized and

        .. math:: weight_{new} \cdot bias_{new} = weight \cdot bias

        is still an orthogonal support vector.
        Two equivalent representations will yield the same such
        normalized embedding.

        :returns: Equivalent embedding where the weight of the output embedding
            is normalized and, when flattened, the weight's first non-zero
            entry is positive
        :raises: :py:exc:`ValueError`, if the weight of the embedding is zero
        """
        # Normalize the weight and the bias
        normed_emb = self.normalize()

        # Now reflect weight if it is on the wrong side of the hemisphere
        # Sign of first non-zero entry
        first_weight_sign = np.sign(
            [entry for entry in normed_emb.normal_vec.reshape(-1)
             if entry != 0][0])
        normed_emb.support_factor *= first_weight_sign
        normed_emb.normal_vec *= first_weight_sign
        normed_emb.scaling_factor *= first_weight_sign

        return normed_emb

    def to_pos_scaling(self) -> 'ConceptEmbedding':
        """Return the representation of this embedding with positive scaling."""
        sign = 1. if self.scaling_factor >= 0 else -1.
        return ConceptEmbedding(normal_vec=sign * self.normal_vec,
                                support_factor=sign * self.support_factor,
                                scaling_factor=sign * self.scaling_factor,
                                **self.meta_info)

    def forget_scaling(self) -> 'ConceptEmbedding':
        """Return the embedding with the same normal vec and support but
        scaling factor 1."""
        return ConceptEmbedding(normal_vec=self.normal_vec,
                                support_factor=self.support_factor,
                                scaling_factor=1.,
                                **self.meta_info)

    @classmethod
    def mean(cls, embeddings: Sequence['ConceptEmbedding']):
        r"""Get the normalized embedding with distance fctn mean of the
        normalized distance fctns.
        Consider the non-scaled distance functions of the normalized versions
        of the given embeddings. Then the condition for the normalized mean
        embedding is that at any point the distance from the embedding
        hyperplane to the point is the mean distance of these normalized
        distances:

        .. math::
            d_{\frac{n}{|n|}, b\cdot |n|}
            = mean\left( d_{\frac{n_j}{|n_j|}, |n_j|\cdot b_j} \right)

        The scaling factor in the end is the mean of the scaling factors of
        the normalized representations of the given embeddings.

        :returns: normalized
        :raises: :py:exc:`ValueError` if
            :math:`mean\left(\frac{n_j}{|n_j|}\right)` of the scaled
            normal vectors :math:`n_j` is 0
        """
        # Value checks:
        # Check embeddings length
        cls._validate_embedding_list(embeddings)

        # Get normalized versions of embeddings with positive scaling factor
        normed_embs = [e.normalize().to_pos_scaling() for e in embeddings]
        scaling_factors = [float(e.scaling_factor) for e in normed_embs]
        # Set scaling factors to 1. to make .scale() have no effect
        normed_unscaled_embs: List[ConceptEmbedding] = [e.forget_scaling() for e
                                                        in normed_embs]

        # Now calculate distance mean and normalize normal vector
        mean_embedding: ConceptEmbedding = cls.mean_by_distance(
            normed_unscaled_embs).normalize()
        # The scaling_factor should be the mean of the given scaling factors:
        mean_embedding.scaling_factor = np.mean(scaling_factors)

        return mean_embedding

    @staticmethod
    def _validate_embedding_list(embeddings: Sequence['ConceptEmbedding']
                                 ) -> None:
        """Check whether given embeddings list is suitable for calculating a
        mean.

        :raises: :py:exc:`ValueError` if any condition is not fulfilled.
        """
        # Check embeddings length
        if len(embeddings) == 0:
            raise ValueError("Got empty list of embeddings for calculating "
                             "mean.")
        # Check that all use the same concept
        concept: Concept = embeddings[0].concept
        for emb in embeddings:
            if not concept.name == emb.concept.name:
                raise ValueError(("Called mean on embeddings of different "
                                  "concepts ({} and {})"
                                  ).format(concept.name, emb.concept.name))

    @classmethod
    def mean_by_distance(cls,  # TODO: optional weighting?
                         embeddings: Sequence['ConceptEmbedding']
                         ) -> 'ConceptEmbedding':
        r"""Get embedding with distance measure being the mean of given embs.
        This routine only works if the mean of the scaled embeddings normal
        vectors is non-zero.

        The distance of a point :math:`x` from a hyperplane :math:`(n, b)`
        with normal vector :math:`n` and support vector :math:`b\cdot n` is
        defined as

        .. math::
            d_{n,b}(x)
            = \left((x - b\cdot n) \circ n\right)
            = x \circ n - b \cdot |n|^2

        For an embedding :math:`(n, b, s)` with scaling factor s the distance
        measure is the one of its scaled version :math:`(s n, \frac{b}{s}, 1)`,
        which turns out to be

        .. math:: d_{s n, \frac{b}{s}} = s \cdot d_{n,b}

        This routine determines the "average" hyperplane for the given
        embeddings, where here average hyperplane :math:`(n, b)` means the
        one with the following property:

        .. math::
            d_{n,b}
            = mean(d_{n_j,b_j})
            = \frac 1 N \sum_{j=1}^{N} d_{n_j,b_j}

        i.e. at any point :math:`x` in space the distance of the average
        hyperplane to :math:`x` is the mean of the distances of all N given
        hyperplanes :math:`(n_j,b_j)` to :math:`x`. It is unique (the points
        on the plane are those with distance 0 and thus all the same),
        and given by the following combination (with scaling factor 1):

        .. math::
            n &= mean(n_j) \\
            b &= \frac{1}{|n|^2} mean(b_j \cdot |n_j|^2)

        Possible problems: This will weight the contribution of the given
        embeddings by their confidence, i.e. their scaling factor.
        To avoid this, the mean can be taken over the normalized versions
        with scaling factor set to one and the scaling factor of the mean can
        be determined by confidence calibration.

        :returns: embedding describing the hyperplane with above properties
        :raises: ValueError if the mean of the scaled normal vectors of the
            given embeddings is 0
        """
        # Value checks:
        cls._validate_embedding_list(embeddings)
        meta_info: Dict[str, Any] = embeddings[0].meta_info

        # First apply the scaling to all embeddings
        scaled_embeddings: List[ConceptEmbedding] = \
            [e.scale() for e in embeddings]
        normal_vecs: List[np.ndarray] = \
            [e.normal_vec for e in scaled_embeddings]
        support_factors: List[np.ndarray] = \
            [e.support_factor for e in scaled_embeddings]

        # Normal vector: mean(n_j) with n_j scaled normal vectors
        mean_normal_vec: np.ndarray = np.mean(normal_vecs, axis=0)

        # Get normal vector norm; must not be zero!
        squared_mean_normal_vec_norm: float = float(
            np.sum(mean_normal_vec * mean_normal_vec))
        if np.allclose(squared_mean_normal_vec_norm, 0):
            raise ValueError("Mean of scaled embedding normal vectors is zero; "
                             "cannot calculate mean embedding")

        # Support factor: - b = mean(b_j * |n_j|**2) / (|n|**2)
        # with b_j scaled support factor, n mean normal vector
        mean_support_factor: np.ndarray = \
            (np.mean([b * np.sum(n_j * n_j)
                      for b, n_j in zip(support_factors, normal_vecs)])
             / squared_mean_normal_vec_norm)

        mean_embedding: ConceptEmbedding = cls(
            normal_vec=mean_normal_vec,
            support_factor=mean_support_factor,
            scaling_factor=1.,
            **meta_info
        )
        return mean_embedding

    @classmethod
    def variance(cls, embeddings: Sequence['ConceptEmbedding'], ddof: int = 1
                 ) -> Tuple[np.ndarray, float, float]:
        r"""Get the variance of a list of embeddings (by default unbiased).
        The variances are calculated on the unique normalized representations
        of the embeddings, and encompass variance of:

        - the normal vector
        - the support vector factor (= distance to 0)
        - the scaling factor (= length of the normal vector).

        :param embeddings: sequence of embeddings to take variance of
        :param ddof: delta degrees of freedom: the divisor used in calculations
            is :math:`\text{num_embeddings} - \text{ddof}`;
            if ``ddof=1`` (default), the unbiased variance is obtained
        :returns: Tuple of variance of
            ``(normal vecs, support factors, scaling factors)`` for
            normalized representations of given embeddings
        """
        # First norm all embeddings and bring them to the same hemisphere to
        # compare them
        normed_embs = [e.normalize().to_pos_scaling() for e in embeddings]

        # Now calculate the variances of the embedding specifiers
        var_normal_vec: np.ndarray = \
            np.var([e.normal_vec for e in normed_embs], axis=0, ddof=ddof)
        var_supp_factor: float = float(
            np.var([e.support_factor for e in normed_embs], ddof=ddof))
        var_scale_factor: float = float(
            np.var([e.scaling_factor for e in normed_embs], ddof=ddof))

        return var_normal_vec, var_supp_factor, var_scale_factor

    @classmethod
    def std_deviation(cls, embeddings: Sequence['ConceptEmbedding'],
                      ddof: int = 1
                      ) -> Tuple[np.ndarray, float, float]:
        r"""Get the (by default unbiased) standard deviation of a list of embs.
        The standard deviations are calculated on the unique normalized
        representations of the embeddings, and encompass standard deviation of:

        - the normal vector
        - the support vector factor (= distance to 0)
        - the scaling factor (= length of the normal vector).

        The deviations are calculated as the square root of the variances
        (see :py:meth:`variance`).

        :param embeddings: sequence of embeddings
        :param ddof: delta degrees of freedom: the divisor used in calculations
            is :math:`\text{num_embeddings} - \text{ddof}`;
            if ``ddof=1`` (default), the unbiased standard  deviation is
            obtained
        :returns: Tuple of standard deviation of
            ``(normal vecs, support factors, scaling factors)`` for
            normalized representations of given embeddings
        """
        var_normal_vec, var_supp_factor, var_scale_factor = \
            cls.variance(embeddings, ddof=ddof)
        return (np.sqrt(var_normal_vec),
                np.sqrt(var_supp_factor),
                np.sqrt(var_scale_factor))

    @classmethod
    def mean_by_angle(cls, embeddings: Sequence['ConceptEmbedding']):
        r"""Get embedding where distance to the given hyperplanes at each
        point sums up to 0.

        **The Math Behind**

        This routine approximates an "average" hyperplane from the given
        embeddings where here average hyperplane means the one for which the
        following holds:
        Given a point :math:`x` on the average hyperplane, the signed
        distances to all hyperplanes along the average hyperplane's normal
        vector sum up to zero.
        The signed distance from :math:`x` to a hyperplane H non-orthogonal
        to the average hyperplane is

        .. math::
            \left(\left( (R\cdot n + x) \cap H \right) - x \right) \circ n,

        where

         - :math:`n` is the normalized normal vector of the average hyperplane,
         - :math:`(R \cdot n + x)` is the 1-dim affine sub-space through
           :math:`x` in the direction of :math:`n`, and
         - :math:`((R \cdot n + x) \cap H)` is the unique intersection of
           above line with :math:`H`.

        The average hyperplane has the following properties:

        - The average hyperplane is unique.
        - The average normal vector only depends on the normal vectors of the
          hyperplanes, not their supports/biases.
        - Given the normalized normal vector n of the average hyperplane,
          a support vector is given by:

          .. math::
            \frac{1}{N} \sum_{j=1}^{N} \frac{|b_j|^2}{n \circ b_j} \cdot n

          where the sum goes over the N hyperplanes, :math:`n` is a normalized
          normal vector of the average hyperplane and :math:`b_j` is the
          orthogonal support vector of the jth hyperplane
          (i.e. a support vector which is a multiple of the normal vector).
        - Assume normalized normal vectors of the hyperplanes which all lie in
          the same hypersphere and are given in angle coordinates of the
          1-hypersphere. An entry in the average normal vector in angle
          coordinates is the mean of the entries in the other hyperplane's
          normal vectors.

        **Implementation Notes**

        Normal vector:
          The normal vector is computationally expensive to calculate
          (should be the spherical barycenter of the normed normal vectors
          in one hemisphere)
          and can be approximated by the normalized barycenter of the
          normalized normal vectors which lie in the same hemisphere.

        Support:
          If the normal vectors do not differ too much, the support can also
          be approximated by the mean of the orthogonal support vectors
          (or be considered as an optimisation problem
          and be learned from the concept data).

        :returns: The embedding representing the average hyperplane of the
            hyperplanes represented by the given embeddings
        :raises: :py:exc:`ValueError` if the mean of the normalized normal
            vectors :math:`\frac{n_j}{|n_j|}` of the given embeddings is 0
        """
        # Value checks:
        cls._validate_embedding_list(embeddings)
        meta_info: Dict[str, Any] = embeddings[0].meta_info

        # First norm all embeddings and bring them to the same hemisphere
        normed_embeddings = [e.normalize().to_pos_scaling() for e in embeddings]
        normed_normal_vecs = [e.normal_vec for e in normed_embeddings]
        normed_support_factors = [e.support_factor for e in normed_embeddings]
        normed_scaling_factors = [e.scaling_factor for e in normed_embeddings]

        # Approximate normal vector: Take normalized mean of the concept vectors
        mean_normal_vec: np.ndarray = np.mean(normed_normal_vecs, axis=0)
        mean_normal_vec_length = np.linalg.norm(mean_normal_vec)
        if np.allclose(mean_normal_vec_length, 0):
            raise ValueError("Mean of normalized embedding normal vectors is "
                             "zero (all positive scaling); "
                             "cannot calculate mean embedding")
        mean_normal_vec /= mean_normal_vec_length
        mean_scaling_factor: np.ndarray = np.mean(normed_scaling_factors,
                                                  axis=0)

        # Support bias:
        # 1/N \sum_{j=1}^{N} bias_j^2 / (mean_weight \circ (bias_j * weight_j))
        cos_distances = [np.sum(mean_normal_vec * (bias * weight))
                         for bias, weight in
                         zip(normed_support_factors, normed_normal_vecs)]
        mean_support_factor = ((1 / len(embeddings)) *
                               # supp / cos(supp to norm_weight)
                               np.sum([
                                   np.abs(bias) ** 2 / cos_dist
                                   if cos_dist != 0 else 0
                                   # no or orthogonal support
                                   for bias, cos_dist in
                                   zip(normed_support_factors, cos_distances)
                               ]))

        mean_embedding: ConceptEmbedding = cls(
            normal_vec=mean_normal_vec,
            support_factor=mean_support_factor,
            scaling_factor=mean_scaling_factor,
            **meta_info
        )
        return mean_embedding

    def __eq__(self, other: 'ConceptEmbedding'):
        """Convert both embeddings to unique representation and compare values.
        """
        norm_other = other.normalize().to_pos_scaling()
        norm_self = self.normalize().to_pos_scaling()
        return (np.allclose(norm_other.scaling_factor,
                            norm_self.scaling_factor)) and \
               (np.allclose(norm_other.support_factor,
                            norm_self.support_factor)) and \
               (np.allclose(norm_other.normal_vec, norm_self.normal_vec))

    def __repr__(self) -> str:
        """Information about concept, model, layer, concept vector and thresh.
        """
        return (("{cls_name}("
                 "concept={concept_name}, "
                 "layer_id={layer_id}, "
                 "normal_vec={normal_vec}, "
                 "support_factor={support_factor}, "
                 "scaling_factor={scaling_factor}, "
                 "model={model})")
                .format(cls_name=self.__class__.__name__,
                        concept_name=self.concept_name,
                        layer_id=self.layer_id,
                        normal_vec=self.normal_vec,
                        support_factor=self.support_factor,
                        scaling_factor=self.scaling_factor,
                        model=self.main_model))

    def __str__(self):
        return repr(self)
