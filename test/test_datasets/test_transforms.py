"""Tests for dataset modifiers."""
#  Copyright (c) 2020 Continental Automotive GmbH

# pylint: disable=not-callable
# pylint: disable=no-member
# pylint: disable=no-self-use
from collections import namedtuple
from typing import Tuple, Set, Dict, List

import PIL.Image
import numpy as np
import pytest
import torch

from hybrid_learning.datasets.transforms import Binarize, BatchIoUEncode2D, \
    BatchIntersectDecode2D, Merge, AND, OR, NOT, PadAndResize, IoUEncode


def test_binarizer():
    """Test binarizing functionality"""
    binarizer: Binarize = Binarize(0.5)
    # Does __str__ and __repr__ work?
    _, _ = str(binarizer), repr(binarizer)
    assert float(binarizer(torch.tensor(3))) == 1
    assert float(binarizer(torch.tensor(0.5))) == 0
    assert float(binarizer(torch.tensor(-1))) == 0


class TestIoUEncoding:
    """Test IoU encoding and decoding functions."""

    IOU_ENCODING_SAMPLES = [
        ([[[[1]]]], [[1]], [[[[1]]]]),
        ([[[[2]]]], [[1]], [[[[2]]]]),
        ([[[[1]]], [[[2]]]], [[1]], [[[[1]]], [[[2]]]]),
        ([[[[1, 1], [1, 1]]]], [[1]], [[[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]]]),
        ([[[[1, 1], [1, 1]]], [[[1, 1], [1, 1]]]], [[1]],
         [[[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]],
          [[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]]]),
        ([[[[1, 1], [1, 0]]]], [[1, 1], [1, 1]],
         [[[[3 / 4, 1 / 6], [1 / 6, 1e-8]]]]),
        ([[[[1, 1], [1, 0]]]], [[0, 1], [1, 1]],
         [[[[1 / 2, 1e-8], [1e-8, 1e-8]]]]),
    ]
    """Some examples of IoU calculation; format: tuples of
    ``(input, proto shape, expected output)``;
    The input requires a size of (1,1, width, height)
    (i.e. respect batch and channel axes),
    the proto_shape requires a size of (kernel width, kernel height).
    """
    BINARIZED_IOU_ENCODING_SAMPLES = [
        ([[[[1]]], [[[2]]]], [[1]], [[[[1]]], [[[1]]]]),
        ([[[[0.51, 1], [2, 1]]]], [[1]], [[[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]]]),
        ([[[[1.1, 0.5001], [10, 20]]], [[[1, 1], [1, 1]]]], [[1]],
         [[[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]],
          [[[1 / 4, 1 / 4], [1 / 4, 1 / 4]]]]),
        ([[[[1, 1], [0.6, 0.3]]]], [[1, 1], [1, 1]],
         [[[[3 / 4, 1 / 6], [1 / 6, 1e-8]]]]),
        ([[[[1, 1], [1, 0.5]]]], [[0, 1], [1, 1]],
         [[[[1 / 2, 1e-8], [1e-8, 1e-8]]]]),
    ]
    """Examples of IoU encoding as in IOU_ENCODING_SAMPLES, but assuming a
    pre_thresh of 0.5"""

    IOU_DECODING_SAMPLES = [
        ([[[[1]]]], [[1]], [[[[1]]]]),
        ([[[[1]]], [[[1]]]], [[1]], [[[[1]]], [[[1]]]]),
        ([[[[1, 1], [1, 1]]]], [[1, 1], [1, 1]], [[[[0.25, 0.5], [0.5, 1]]]]),
        ([[[[0, 0], [0, 1]]]], [[1, 1], [1, 1]], [[[[0, 0], [0, 0.25]]]]),
        ([[[[0, 1], [1, 1]]]], [[0, 1], [1, 1]], [[[[0, 0], [0, 2 / 3]]]]),
    ]
    """Examples of IoU encoding as in IOU_ENCODING_SAMPLES."""

    def test_iou_encoder(self):
        """Test BatchIoUEncode2D"""
        with torch.no_grad():
            # Value checks:
            sample_mask, proto_shape, iou_mask = self.IOU_ENCODING_SAMPLES[0]
            iou_enc = BatchIoUEncode2D(proto_shape=np.array(proto_shape))

            # Do __repr__ and __str__ work?
            _, _ = str(iou_enc), repr(iou_enc)

            # Mask of wrong type should not be accepted
            with pytest.raises(ValueError):
                iou_enc(sample_mask)
            # Mask without batch dimension should not be accepted
            with pytest.raises(ValueError):
                iou_enc(torch.tensor(sample_mask[0]))
            # Mask with wrong channel dimension should not be accepted
            with pytest.raises(ValueError):
                iou_enc(torch.tensor([[sample_mask[0][0], sample_mask[0][0]]]))

            for sample_mask, proto_shape, iou_mask in self.IOU_ENCODING_SAMPLES:
                sample_mask_t: torch.Tensor = torch.tensor(sample_mask).float()
                iou_mask_t: torch.Tensor = torch.tensor(iou_mask).float()
                iou_enc = BatchIoUEncode2D(proto_shape=np.array(proto_shape))
                assert np.all(iou_enc.proto_shape == proto_shape)

                # simple 1x1 proto shape and 1x1 mask:
                iou_enc_mask: torch.Tensor = iou_enc(sample_mask_t)
                assert iou_enc_mask.allclose(iou_mask_t), \
                    ("Wrong IoU output:\noriginal mask: {}\nproto shape: {}"
                     "\niou mask: {}\n{}").format(sample_mask_t, proto_shape,
                                                  iou_enc_mask, repr(iou_enc))

    def test_iou_decoder(self):
        """Test BatchIntersectDecode2D"""
        with torch.no_grad():
            # Value checks on init:
            sample_mask, proto_shape, non_iou_mask = \
                self.IOU_ENCODING_SAMPLES[0]
            iou_dec = BatchIntersectDecode2D(proto_shape=np.array(proto_shape))

            # Do __repr__ and __str__ work?
            _, _ = str(iou_dec), repr(iou_dec)

            # Mask of wrong type should not be accepted
            with pytest.raises(ValueError):
                iou_dec(sample_mask)
            # Mask without batch dimension should not be accepted
            with pytest.raises(ValueError):
                iou_dec(torch.tensor(sample_mask[0]))
            # Mask with wrong channel dimension should not be accepted
            with pytest.raises(ValueError):
                iou_dec(torch.tensor([[sample_mask[0][0], sample_mask[0][0]]]))

            for sample_mask, proto_shape, non_iou_mask in \
                    self.IOU_DECODING_SAMPLES:
                sample_mask_t: torch.Tensor = torch.tensor(sample_mask).float()
                mask_t: torch.Tensor = torch.tensor(non_iou_mask).float()
                iou_dec: BatchIntersectDecode2D = BatchIntersectDecode2D(
                    proto_shape=np.array(proto_shape))
                iou_enc: BatchIoUEncode2D = BatchIoUEncode2D(
                    proto_shape=np.array(proto_shape))

                # The kernel should be the rotated proto type:
                enc_kernel: np.ndarray = \
                    ((iou_enc.intersect_encoder.intersect_conv.weight.data
                      ).detach().cpu().numpy()[0, 0, ...])
                dec_kernel: np.ndarray = iou_dec.decoder_conv \
                    .weight.data.detach().cpu().numpy()[0, 0, ...]
                assert np.allclose(
                    dec_kernel,
                    np.fliplr(np.flipud(enc_kernel / np.sum(enc_kernel)))), \
                    (("Wrong kernel: was\n{} but should have been 180° "
                      "rotation of\n{} which is\n{}"
                      ).format(dec_kernel.tolist(),
                               (enc_kernel / np.sum(enc_kernel)).tolist(),
                               np.fliplr(
                                   np.flipud(enc_kernel / np.sum(enc_kernel))
                               ).tolist()))
                # The padding should be flipped compared to encoding:
                enc_padding: Tuple[int, ...] = \
                    iou_enc.intersect_encoder.padding.padding
                dec_padding: Tuple[int, ...] = \
                    iou_dec.padding.padding
                assert enc_padding == (dec_padding[1], dec_padding[0],
                                       dec_padding[3], dec_padding[2])

                # simple 1x1 proto shape and 1x1 mask:
                iou_dec_mask_t: torch.Tensor = iou_dec(sample_mask_t)
                assert iou_dec_mask_t.allclose(mask_t), \
                    (("Wrong IoU dec output:\noriginal mask: {}\nproto shape: "
                      "{}\niou mask: {}\n{}"
                      ).format(sample_mask_t, proto_shape, iou_dec_mask_t,
                               repr(iou_dec)))

    def test_iou_coders_size(self):
        """Check that the size does not change when IoU decoding/encoding."""
        proto_shape = np.ones((25, 75))
        mask_size = (1, 1, 512, 363)
        mask_t = torch.zeros(mask_size)

        # The size of the mask should not change for decoding:
        iou_dec = BatchIntersectDecode2D(proto_shape=proto_shape)
        assert iou_dec(mask_t).size() == mask_size, "IoU decoding changes size!"

        # The size of the mask should not change for encoding:
        iou_enc = BatchIoUEncode2D(proto_shape=proto_shape)
        assert iou_enc(mask_t).size() == mask_size, "IoU encoding changes size!"

    def test_iou_encoder_with_thresh(self):
        """Test IoU encoding of masks with thresholds."""
        with torch.no_grad():
            # check default proto_shape
            kernel_size = (1, 1)
            # noinspection PyTypeChecker
            iou_wrap = IoUEncode(kernel_size=kernel_size)
            # Do __repr__ and __str__ work?
            _, _ = str(iou_wrap), repr(iou_wrap)
            assert np.allclose(iou_wrap.proto_shape, kernel_size)

            for pre_thresh, samples in (
                    # without binarizing, targets should just be IoU encoded
                    (None, self.IOU_ENCODING_SAMPLES),
                    # examples with pre-binarizing (post-binarizing is tested
                    # via binarize tests)
                    (0.5, self.BINARIZED_IOU_ENCODING_SAMPLES)):
                for sample_mask, proto_shape, iou_mask in samples:
                    sample_mask_t: torch.Tensor = \
                        torch.tensor(sample_mask).float()
                    iou_mask_t: torch.Tensor = torch.tensor(iou_mask).float()
                    iou_wrap = IoUEncode(proto_shape=proto_shape,
                                         pre_thresh=pre_thresh,
                                         kernel_size=kernel_size,
                                         batch_wise=True)

                    # The kernel size is overridden by proto_shape:
                    assert np.allclose(iou_wrap.proto_shape, proto_shape)

                    targets = iou_wrap(sample_mask_t)
                    assert targets.allclose(iou_mask_t), \
                        (("Wrong IoU output:\norig targets: {}\nproto shape: {}"
                          "\npredicted iou targets: {}\nexpected iou targets: "
                          "{}\ncontext: {}")
                         .format(sample_mask_t, proto_shape, targets,
                                 iou_mask_t, repr(iou_wrap)))

    # noinspection PyTypeChecker
    def test_iou_decoder_with_thresh(self):
        """Test IoU decoding of masks."""
        with torch.no_grad():
            # check default proto_shape
            kernel_size = (1, 1)
            iou_wrap = IoUEncode(kernel_size=kernel_size)

            # Do __repr__ and __str__ work?
            _, _ = str(iou_wrap), repr(iou_wrap)

            assert np.allclose(iou_wrap.proto_shape, kernel_size)


class TestMergeOperations:
    """Test the dictionary merge operations."""

    def test_equals(self):
        """Test __eq__"""
        equal_samples = (
            (AND('a', 'b'), AND('b', 'a')), (OR('a', 'b'), OR('b', 'a')),
            (AND(NOT('a'), 'b'), AND('b', NOT('a'))),
        )
        for first, second in equal_samples:
            assert first == second

        not_equal_samples = (
            (AND(NOT('a'), 'b'), AND('a', NOT('b'))),
        )
        for first, second in not_equal_samples:
            assert first != second

    def test_normalized_repr(self):
        """Test the normalized_repr function"""
        samples = (
            ("b&&a", "a&&b"), ("b||a", "a||b"),
            ("a&&b", "a&&b"), ("a||b", "a||b"), ("~a", "~a"),
            ("b&&d||~c||~e&&a", "a&&b&&d||~c||~e")
        )
        for before, after in samples:
            assert str(Merge.parse(before).normalized_repr()) == after

    def test_repr_and_str(self):
        """Test the string and representation function."""
        samples = (
            (AND('a', 'b'), "a&&b", "AND('a', 'b')"),
            (AND('b', 'a'), "b&&a", "AND('b', 'a')"),
            (OR('a', 'b'), "a||b", "OR('a', 'b')"),
            (OR('b', 'a'), "b||a", "OR('b', 'a')"),
            (NOT('a'), "~a", "NOT('a')"),
            (AND(NOT('a'), OR('b', NOT('c'))), "~a&&b||~c",
             "AND(NOT('a'), OR('b', NOT('c')))"),
            (AND('a', 'b', out_key='c'), "a&&b", "AND('a', 'b', out_key='c')")
        )
        for item, str_out, repr_out in samples:
            assert str(item) == str_out
            assert repr(item) == repr_out
            assert item == Merge.parse(str_out)

    def test_parse(self):
        """Test main properties of parsing like idempotence"""
        # parsing and usual calling interchangeable
        samples = (
            ("a&&b", AND('a', 'b')), ("a||b", OR('a', 'b')), ("~c", NOT('c')),
            ("a&&b||c", AND('a', OR('b', 'c'))),
        )
        for before, after in samples:
            assert Merge.parse(before) == after

        # idempotence
        formulas = ["a&&b", "a||b", "~a", "a&&b||~c"]
        for formula in formulas:
            assert str(Merge.parse(formula)) == formula

        # out_key option
        assert Merge.parse("b&&a", out_key="c").out_key == "c"
        assert not Merge.parse("b&&a", overwrite=False).overwrite

        # wrong formulas
        with pytest.raises(ValueError):
            Merge.parse("~~a")
        with pytest.raises(ValueError):
            Merge.parse("&&||")
        with pytest.raises(ValueError):
            Merge.parse("~")

    def test_init(self):
        """Test the different init arguments and the
        is_conjunctive_normal_form check."""
        # more than one input value
        AND('a', 'b', 'c')
        OR('a', 'b', 'c')

        # out_key, overwrite
        assert not AND('a', 'b', overwrite=False).overwrite
        assert not OR('a', 'b', overwrite=False).overwrite
        assert not NOT('a', overwrite=False).overwrite
        assert AND('a', 'b', out_key='c').out_key == 'c'
        assert OR('a', 'b', out_key='c').out_key == 'c'
        assert NOT('a', out_key='c').out_key == 'c'

        # NOT may not have more than one input
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            # pylint: disable=too-many-function-args
            NOT('a', 'b')
            # pylint: enable=too-many-function-args
        # NOT must have string input
        for non_str in (AND('a', 'b'), OR('a', 'b')):
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                NOT(non_str)

        # AND, OR must have >= 2 entries
        for invalid_args in ([], ['a'], [NOT('a')]):
            with pytest.raises(ValueError):
                AND(*invalid_args)
            with pytest.raises(ValueError):
                OR(*invalid_args)

        # Not conjunctive normal form
        # AND inside of another operation
        with pytest.raises(ValueError):
            OR(AND('a', 'b'), 'c')
        with pytest.raises(ValueError):
            AND(AND('a', 'b'), 'c')
        # OR inside of OR
        with pytest.raises(ValueError):
            OR(OR('a', 'b'), 'c')

    def test_call(self):
        """Test the __call__ function on some samples"""
        # don't overwrite
        for spec in ("a&&b", "a||b", "~a"):
            with pytest.raises(KeyError):
                Merge.apply(spec, {'c': False}, out_key='c', overwrite=False)

        # overwrite when specified
        out: Dict[str, float] = Merge.apply("a&&b", {'a': 1, 'b': 0},
                                            out_key='a', overwrite=True)
        assert out['a'] == 0

        # masks do not coincide in size
        for spec in ("a&&b", "a||b"):
            with pytest.raises(ValueError):
                Merge.apply(spec, {'a': np.ones(3), 'b': np.zeros(2)})

        size = [3, 3]
        # samples in format (spec, annotations, out)
        samples = (
            # simple examples
            ("a&&b", {"a": np.ones(size), "b": np.zeros(size)}, np.zeros(size)),
            ("a||b", {"a": np.ones(size), "b": np.zeros(size)}, np.ones(size)),
            ("~a", {"a": np.ones(size)}, np.zeros(size)),
            # more sophisticated ones
            ("a&&b", {"a": np.array([0, 1, 0]), "b": np.array([1, 0, 1])},
             np.zeros(size)),
            ("a||b", {"a": np.array([0, 1, 0]), "b": np.array([1, 0, 1])},
             np.ones(size)),
            ("~a", {"a": np.array([0, 1, 0])}, np.array([1, 0, 1])),
            # mix of scalar and mask
            ("a&&b", {"a": np.array([0, 1, 0]), "b": 1}, np.array([0, 1, 0])),
            ("a||b", {"a": np.array([0, 1, 0]), "b": 1}, np.ones(size)),
            ("~a", {"a": True}, False),
            ("a&&~b", {"a": np.array([0, 1, 0]), "b": 0}, np.array([0, 1, 0])),
            ("a||~b", {"a": np.array([0, 1, 0]), "b": 0}, np.ones(size)),
            # more than 2 values
            ("a&&b&&c||b",
             {"a": np.array([0, 1, 1, 1]), "b": np.array([1, 0, 1, 1]),
              "c": np.array([1, 1, 0, 1])}, np.array([0, 0, 1, 1])),
        )

        for spec, ann, out in samples:
            orig_keys: Set[str] = set(ann.keys())
            operat: Merge = Merge.parse(spec, out_key='out')
            out_dict = operat(ann)
            # new keys added
            assert {*orig_keys, *operat.all_out_keys} == {*out_dict}, \
                "op: {}".format(repr(operat))
            # other values not changed
            for k in ann:
                assert out_dict[k] is ann[k]
            # correct out value
            assert np.allclose(out_dict['out'], out), \
                "op: {}".format(repr(operat))

    def test_properties(self):
        """Test the properties around in_keys."""
        key_spec = namedtuple("KeySpec",
                              ['spec', 'children', 'consts', 'operation_keys',
                               'all_in_keys', 'all_out_keys'])
        samples: List[key_spec] = [
            key_spec(spec="a&&~b", children=["~b"], consts={"a"},
                     operation_keys={'a', '~b'},
                     all_in_keys={'a', 'b'}, all_out_keys={'a&&~b', '~b'}),
            key_spec(spec="a||b&&~c&&d", children=["a||b", "~c"], consts={"d"},
                     operation_keys={'a||b', '~c', 'd'},
                     all_in_keys={'a', 'b', 'c', 'd'},
                     all_out_keys={'a||b', '~c', 'a||b&&~c&&d'})
        ]

        for spec in samples:
            operat = Merge.parse(spec.spec)
            assert operat.children == [Merge.parse(c) for c in spec.children]
            assert operat.consts == spec.consts
            assert operat.operation_keys == spec.operation_keys
            assert operat.all_in_keys == spec.all_in_keys
            assert operat.all_out_keys == spec.all_out_keys

        # children
        # consts
        # operation keys
        # all_in_keys
        # all_out_keys


def test_pad_and_resize():
    """Test image PadAndResize."""
    trafo: PadAndResize = PadAndResize(img_size=(6, 6),
                                       interpolation=PIL.Image.NEAREST)
    assert trafo.img_size == (6, 6)

    img_t: torch.Tensor = torch.ones(size=(1, 1, 3), device='cpu')
    transformed: torch.Tensor = trafo(img_t)
    expected: np.ndarray = np.zeros((1, *trafo.img_size))
    expected[:, 2:4, :] = 1
    assert isinstance(transformed, torch.Tensor)
    assert transformed.size()[1:] == trafo.img_size
    assert transformed.numpy().shape == expected.shape
    assert np.allclose(transformed.numpy(), expected), \
        "Transformed array:\n{}\nExpected array:\n{}".format(
            transformed.numpy(), expected
        )
