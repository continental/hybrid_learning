"""Tests for dataset modifiers."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=not-callable
# pylint: disable=no-member
# pylint: disable=no-self-use
from typing import Tuple, List, Sequence, Optional

import numpy as np
import pytest
import torch
from hybrid_learning.datasets import transforms as trafos

from hybrid_learning.datasets.transforms import \
    Binarize, PadAndResize, IoUEncode, ToActMap, ToBBoxes, Threshold, \
    general_add, Identity, Compose, SameSize, TupleTransforms, ReduceTuple, \
    ToTensor
from hybrid_learning.datasets.transforms.encoder import \
    BatchIoUEncode2D, BatchIntersectDecode2D, BatchBoxBloat


def test_binarizer():
    """Test binarizing functionality"""
    # noinspection PyArgumentEqualDefault
    binarizer: Binarize = Binarize(0.5)
    # Does __str__ and __repr__ work?
    _, _ = str(binarizer), repr(binarizer)

    # Some example values
    assert float(binarizer(torch.tensor(3))) == 1
    assert float(binarizer(torch.tensor(0.5))) == 0
    assert float(binarizer(torch.tensor(-1))) == 0


def test_threshold():
    """Test the simple thresholding transformation."""
    # Set lower bound constant:
    thresholder: Threshold = Threshold(0.5, 1., None)
    assert float(thresholder(torch.tensor(3))) == 3
    assert float(thresholder(torch.tensor(1))) == 1
    assert float(thresholder(torch.tensor(-1))) == 1
    assert float(thresholder(torch.tensor(0.25))) == 1
    assert float(thresholder(torch.tensor(0.5))) == 1

    # Set upper bound constant:
    thresholder: Threshold = Threshold(0.5, None, 1.)
    assert float(thresholder(torch.tensor(3))) == 1
    assert float(thresholder(torch.tensor(1))) == 1
    assert float(thresholder(torch.tensor(-1))) == -1
    assert float(thresholder(torch.tensor(0.25))) == 0.25
    assert float(thresholder(torch.tensor(0.5))) == 0.5

    # Setting no high and low values isn't useful but shouldn't raise:
    thresholder: Threshold = Threshold(0.5, None, None)
    assert float(thresholder(torch.tensor(3))) == 3
    assert float(thresholder(torch.tensor(1))) == 1
    assert float(thresholder(torch.tensor(-1))) == -1

    thresholder: Threshold = Threshold(torch.tensor([0.25, 0.5, 1]), 0, None)
    assert thresholder(torch.tensor([1, 1, 1])).numpy().tolist() == [1, 1, 0]
    assert thresholder(torch.tensor([0.5, 0.25, 0.5])).numpy().tolist() == \
           [0.5, 0, 0]
    assert thresholder(torch.tensor([0.5, 0.25, 3])).numpy().tolist() == \
           [0.5, 0, 3]


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

    def test_nms(self):
        """Test the non-max-suppression bloating."""
        # Identity
        identity = BatchBoxBloat((1, 1))
        assert tuple(identity.padding.padding) == (0, 0, 0, 0)
        # mask batches without channel information
        ex_masks: Tuple[torch.Tensor, ...] = (
            torch.rand((2, 3, 3)),
            torch.ones((1, 1, 1)),
            torch.zeros((3, 5, 5)))
        for masks in ex_masks:
            masks = masks.unsqueeze(1)  # add channel dimension
            output: torch.Tensor = identity(masks)
            assert output.float() is output
            assert masks.detach().numpy().tolist() == \
                   output.detach().numpy().tolist()

        # Some example values
        # Format: Tuple of tuples (kernel_size, mask, exp_output)
        # with mask and expected output without batch and channel dimension
        ex_masks: Tuple[Tuple, ...] = (
            # Just about size preservation
            ([3, 3], [[0] * 3] * 3, [[0] * 3] * 3),
            # 2x1 kernel
            ([2, 1], [[0, 0], [1, 0]], [[1, 0], [1, 0]]),
            # 2x1 kernel
            ([2, 1], [[0.5, 2], [1, 1]], [[1, 2], [1, 1]]),
            # 2x2 kernel
            ([2, 2], [[2, 0.5], [1.7, 1.9]], [[2, 1.9], [1.9, 1.9]]),
            # 2x2 kernel with padding
            ([2, 2], np.arange(9).reshape((3, 3)),
             [[4, 5, 5], [7, 8, 8], [7, 8, 8]]),
        )
        for kernel_size, masks, exp_output in ex_masks:
            nms = BatchBoxBloat(kernel_size)
            masks = torch.tensor(masks).unsqueeze(0).unsqueeze(0)
            masks_size: List[int] = list(masks.size())
            exp_output = torch.tensor(exp_output).unsqueeze(0).unsqueeze(0)
            output = nms(masks)

            # float output
            assert output.float() is output
            # no size change
            assert list(output.size()) == masks_size, \
                ("Size mismatch for kernel_size {}\nin\n{}\nout\n{}"
                 .format(kernel_size, masks, output))
            # correct value
            assert (exp_output.detach().float().numpy().tolist() ==
                    output.detach().numpy().tolist()), \
                ("Wrong out for kernel_size {}\nin\n{}\nexpected\n{}\nout\n{}"
                 .format(kernel_size, masks, exp_output, output))


def test_pad_and_resize():
    """Test image PadAndResize."""
    trafo: PadAndResize = PadAndResize(img_size=(6, 6),
                                       interpolation="nearest")
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

    # Do not confuse height and width:
    # noinspection PyArgumentEqualDefault
    trafo: PadAndResize = PadAndResize(img_size=(6, 8),
                                       interpolation="bilinear")
    transformed: torch.Tensor = trafo(img_t)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.size()[1:] == (6, 8)
    assert transformed.numpy().shape[1:] == (6, 8)


def test_to_act_map():
    """Test the ToActMap transforms."""

    # pylint: disable=abstract-method
    class DummyModule(torch.nn.Module):
        """Dummy torch module that simply adds 1 to each tensor entry."""

        # noinspection PyMethodMayBeStatic
        def forward(self, tens: torch.Tensor):
            """Add 1 to each tensor entry."""
            return tens + 1

    dummy_mod = DummyModule()

    # Invalid device raises error:
    with pytest.raises(RuntimeError):
        ToActMap(act_map_gen=dummy_mod, device="blub")

    # Some value checks using dummy module:
    assert ToActMap(act_map_gen=dummy_mod)(torch.ones(1)) \
        .equal(torch.ones(1) + 1)
    assert ToActMap(act_map_gen=dummy_mod)(torch.tensor([1, 2, 3])) \
        .equal(torch.tensor([2, 3, 4]))


def test_to_bboxes():
    """Test the ToBBoxes transformation."""
    assert repr(ToBBoxes(bbox_size=(1, 2))) == \
           "ToBBoxes(bbox_size=(1, 2))"

    sizes_thresh_masks_expected: List[Tuple[
        Tuple[int, int], float, List, List]] = [
        # Bounding box size of 1x1 --> cannot overlap
        ((1, 1), 0.5, [[1, 0], [0.5, 0]], [[1, 0], [0.5, 0]]),
        # correct padding for even box size:
        ((2, 2), 0.5,
         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        # box ranging out of mask:
        ((2, 2), 0.5,
         [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
         [[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        # correct bbox extends for non-square bbox and mask:
        ((2, 3), 0.5,
         [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
         [[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],),
        # correct overlapping:
        ((2, 2), 0.5,
         [[0, 0, 0], [0, 1, 0], [0, 0, 0.5]],
         [[1, 1, 0], [1, 1, 0.5], [0, 0.5, 0.5]]),
        # correct nms:
        ((2, 2), 0.5,
         [[0, 0, 0], [0, 1, 0], [0, 0, 0.5]],
         [[1, 1, 0], [1, 1, 0.5], [0, 0.5, 0.5]]),
        # correct nms filtering:
        ((2, 3), 0.4,
         [[0, 0, 0, 0], [1, .5, .25, 0], [0, 0, 0, 0]],
         [[1, 1, .25, .25], [1, 1, .25, .25], [0, 0, 0, 0]],),
    ]
    for bbox_size, iou_thr, mask_l, exp_mask_l in sizes_thresh_masks_expected:
        mask = torch.tensor(mask_l, dtype=torch.float)
        exp_mask = torch.tensor(exp_mask_l, dtype=torch.float)
        trafo = ToBBoxes(bbox_size=bbox_size, iou_threshold=iou_thr)
        outp = trafo(mask)
        context = ("trafo: {}\ninput: {}\nexpected: {}\noutput: {}"
                   .format(trafo, mask, exp_mask, outp))
        assert outp.isclose(exp_mask).all(), context


def test_general_add():
    """Test the general addition defined for the transforms."""
    # Copy applied:
    ident: Identity = Identity()
    output = general_add(ident, None, composition_class=Compose,
                         identity_class=Identity)
    assert output == ident
    assert output is not ident

    samples: List[Sequence[Optional[TupleTransforms]]] = [
        # Some examples for TupleTransforms:
        (None, None, None),
        (None, Identity(), Identity()),
        (Identity(), Identity(), Identity()),
        (Identity(), None, Identity()),
        (Identity(), SameSize(), SameSize()),
        (SameSize(), Identity(), SameSize()),
        (Compose([Identity()]), SameSize(), Compose([Identity(), SameSize()])),
        (SameSize(), Compose([Identity()]), Compose([SameSize(), Identity()])),
        (Compose([Identity(), SameSize()]), Compose([Identity(), SameSize()]),
         Compose([Identity(), SameSize(), Identity(), SameSize()])),
        (Identity(), list(), NotImplemented),
        (SameSize(), ReduceTuple(Identity(), torch.add),
         Compose([SameSize(), ReduceTuple(Identity(), torch.add)])),
        # Some examples for ImageTransforms:
        (Compose([Identity()]), PadAndResize((1, 1)),
         Compose([Identity(), PadAndResize((1, 1))])),
        (PadAndResize((1, 1)), Compose([Binarize()]),
         Compose([PadAndResize((1, 1)), Binarize()])),
        (Compose([Threshold()]), Compose([PadAndResize((1, 1)), Binarize()]),
         Compose([Threshold(), PadAndResize((1, 1)), Binarize()])),
        (list(), Binarize(), NotImplemented),
    ]
    for trafo1, trafo2, expected in samples:
        output = general_add(trafo1, trafo2, composition_class=Compose,
                             identity_class=Identity)
        context = dict(t1=trafo1, t2=trafo2, expected=expected, output=output)
        assert repr(output) == repr(expected), \
            "Unexpected sum result; Context:\n{}".format(context)
        assert output == expected, \
            "Unexpected sum result; Context:\n{}".format(context)
        if expected not in (None, NotImplemented):
            plain_sum = (trafo1 + trafo2)
            assert output == plain_sum, \
                "Sums unequal; Context:\n{}".format({**context,
                                                     'plain sum': plain_sum})


def test_to_tensor():
    """Test function for the to sparse tensor helper function."""

    examples = [
        # scalar
        torch.tensor(5),
        # all-zero tensor
        torch.zeros((3, 5, 2)),
        # very sparse tensor of size 2x3
        torch.tensor([[1, 0, 0], [0, 0, 0]]),
        # tensor of size 3x2
        torch.tensor([[1, 3], [0, 2], [0, 0]]),
        # other types
        torch.tensor([[1.3, 0], [0, 2.5]], dtype=torch.float64),
    ]
    for inp in examples:
        context = f"input: {inp}"

        # no modifications
        out: torch.Tensor = ToTensor()(inp)
        assert out.dtype == inp.dtype, f"wrong dtype for {context}"
        assert out.size() == inp.size(), f"size change for {context}"
        assert out.requires_grad == inp.requires_grad, f"requires grad change for {context}"

        # requires grad
        out: torch.Tensor = ToTensor(requires_grad=False)(inp)
        assert out.dtype == inp.dtype, f"wrong dtype for {context}"
        assert out.size() == inp.size(), f"size change for {context}"
        assert not out.requires_grad, f"requires grad not set to False for {context}"

        # sparse
        out: torch.sparse.Tensor = ToTensor(sparse=True)(inp)
        if len(inp.size()) == 0:
            assert not out.is_sparse
            continue
        assert torch.equal(out.to_dense(), inp), f"not invertible for {context}"
        out: torch.sparse.Tensor = ToTensor(sparse=False)(inp)
        assert not out.is_sparse, \
            f"conversion to sparse with sparse set to False for {context}"

        # idempotence
        assert out.eq(ToTensor()(out)).all()
        assert out.eq(ToTensor(sparse=False)(ToTensor(sparse=True)(out))).all()


@pytest.mark.parametrize("nonzero,dim,dtype,sparse_is_better", [
    (0, 1, torch.int, True), (1, 10, torch.int, False),
    (0, 1, torch.float, True), (1, 10, torch.float, False),
    (0.05, 3, torch.int8, False), (0.03, 3, torch.int8, True),
    (0.05, 3, torch.uint8, False), (0.03, 3, torch.uint8, True),
    (0.08, 3, torch.float16, False), (0.07, 3, torch.float16, True),
    (0.08, 3, torch.bfloat16, False), (0.07, 3, torch.bfloat16, True),
    (0.08, 3, torch.int16, False), (0.07, 3, torch.int16, True),
    (0.15, 3, torch.float32, False), (0.14, 3, torch.float32, True),
    (0.15, 3, torch.int32, False), (0.14, 3, torch.int32, True),
    (0.26, 3, torch.float64, False), (0.24, 3, torch.float64, True),
    (0.26, 3, torch.int64, False), (0.24, 3, torch.int64, True),
])
def test_sparse_smaller(nonzero, dim, dtype, sparse_is_better):
    """Test the sparsify check."""
    tens_parts = []
    nonzero_abs = int(nonzero * 100)
    if nonzero_abs > 0:
        tens_parts.append(torch.ones((int(nonzero * 100),)))
    if (100 - nonzero_abs) > 0:
        tens_parts.append(torch.zeros((100 - int(nonzero * 100))))
    tens = torch.cat(tens_parts) if len(tens_parts) > 1 else tens_parts[0]
    tens = tens.view((*([1] * (dim - 1)), tens.numel())).to(dtype)
    print(tens.size(), tens.numel(), tens)

    assert ToTensor.is_sparse_smaller(tens) == sparse_is_better
    assert ToTensor(sparse='smallest')(tens).is_sparse == sparse_is_better


@pytest.mark.parametrize("args,masks,expected,error", [
    # two-tuple restriction
    (dict(resize_target=True), (1,2,3), None, IndexError),
    (dict(only_two_tuples=True), (1,2,3), None, IndexError),
    # empty input
    (None, [], None, IndexError),
    # two-tuple examples
    (dict(resize_to_index=0), [torch.tensor([[1,2],[3,4]]), torch.tensor([[1]])], [torch.tensor([[1,2],[3,4]]), torch.tensor([[1, 1], [1, 1]])], None),
    (dict(resize_target=True), [torch.tensor([[1,2],[3,4]]), torch.tensor([[1]])], [torch.tensor([[1,2],[3,4]]), torch.tensor([[1, 1], [1, 1]])], None),
    (dict(resize_to_index=-1), [torch.tensor([[1,1],[1,1]]), torch.tensor([[1]])], [torch.tensor([[1]]), torch.tensor([[1]])], None),
    (dict(resize_to_index=1), [torch.tensor([[1,1],[1,1]]), torch.tensor([[1]])], [torch.tensor([[1]]), torch.tensor([[1]])], None),
    # three-tuple examples
    (dict(resize_to_index=2), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([[1]])]*3, None),
    (dict(resize_to_index=-1), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([[1]])]*3, None),
    (dict(resize_to_index=1), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([1]*9).view(3,3)]*3, None),
    (dict(resize_to_index=-2), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([1]*9).view(3,3)]*3, None),
    (dict(resize_to_index=0), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([1]*4).view(2,2)]*3, None),
    (dict(resize_to_index=-3), [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([1]*4).view(2,2)]*3, None),
    (None, [torch.tensor([1]*4).view(2,2), torch.tensor([1]*9).view(3,3), torch.tensor([[1]])], [torch.tensor([1]*4).view(2,2)]*3, None),

])
def test_same_size(args, masks, expected, error):
    if error is not None:
        with pytest.raises(error):
            op = SameSize(**(args or {}))
            op(*masks)
    else:
        op = SameSize(**(args or {}))
        masks = [torch.tensor(mask).float() for mask in masks]
        expected = [torch.tensor(mask).float() for mask in expected]
        resized: Tuple[torch.Tensor, ...] = op(*masks)
        # Same number of masks returned:
        assert len(resized) == len(expected)
        # The target size mask is not changed:
        assert resized[op.resize_to_index].allclose(masks[op.resize_to_index])
        # Correct outputs:
        for i in range(len(masks)):
            assert isinstance(resized[i], torch.Tensor)
            assert resized[i].allclose(torch.tensor(expected[i])), \
                "\nExpected: {}\nGot: {}\nAll out masks: {}".format(expected[i], resized[i], resized)
        
        # No-op variant:
        if not op.only_two_tuples:
            op = SameSize(**{**(args or {}), "resize_to_index": 0})
            single_in_out: torch.Tensor = op(masks[0])
            assert isinstance(single_in_out, masks[0].__class__)
            assert single_in_out.allclose(masks[0])

@pytest.mark.parametrize('indices,inp,expected,error', [
    # non-int indices
    ('non-int', None, None, ValueError),
    (1.1, None, None, ValueError),
    # too short tuples
    (2, [False, False], None, IndexError),
    (-3, [False, False], None, IndexError),
    # Single positive index
    (0, [False], [True], None),
    (0, [False, False], [True, False], None),
    (1, [False, False], [False, True], None),
    (0, [False, False, False], [True, False, False], None),
    (1, [False, False, False], [False, True, False], None),
    (2, [False, False, False], [False, False, True], None),
    # Negative index
    (-1, [False, False, False], [False, False, True], None),
    (-2, [False, False, False], [False, True, False], None),
    (-3, [False, False, False], [True, False, False], None),
    # List of indices
    ([0, 1], [False, False, False], [True, True, False], None),
    ([0, 2], [False, False, False], [True, False, True], None),
    ([0, 2, -2], [False, False, False], [True, True, True], None),
    # Do not apply trafo twice
    ([0, 1, -2], [False, False, False], [True, True, False], None),
])
def test_on_index(indices, inp, expected, error):
    trafo = lambda x: not x
    if error is not None:
        with pytest.raises(error):
            op = trafos.OnIndex(indices, trafo)
            op(*inp)
    else:
        op = trafos.OnIndex(indices, trafo)
        assert list(op(*inp)) == expected
        assert trafos.OnAll(trafo)(*inp) == tuple([True] * len(expected))


@pytest.mark.parametrize('indices,inps,expected', [
    # No-op:
    ([0,1,2], [0,1,2], (0,1,2)),
    # Change order:
    ([1,0,2], [0,1,2], (1,0,2)),
    # Subset:
    ([0,1], [0,1,2], (0,1)),
    ([0,2], [0,1,2], (0,2)),
    ([1,2], [0,1,2], (1,2)),
    ([0], [0,1,2], (0,)),
    ([1], [0,1,2], (1,)),
    ([2], [0,1,2], (2,)),
    # Negative indices:
    ([0,-2], [0,1,2], (0,1)),
    ([0,-1], [0,1,2], (0,2)),
    ([-2,-1], [0,1,2], (1,2)),
    ([-3], [0,1,2], (0,)),
    ([-2], [0,1,2], (1,)),
    ([-1], [0,1,2], (2,)),
    # Remove duplicates:
    ([0, 0], [0, 1, 2], (0,)),
    ([1, 1], [0, 1, 2], (1,)),
    ([-1, -1], [0, 1, 2], (2,)),
    ([0, -3], [0, 1, 2], (0,)),
    ([-2, 1], [0, 1, 2], (1,)),
    # Remove duplicates in right order:
    ([0, 1, -3], [0, 1, 2], (0, 1)),
])
def test_subset_tuple(indices, inps, expected):
    assert trafos.SubsetTuple(indices)(*inps) == expected

def test_on_x_twotuple_trafos():
    trafo = lambda x: True
    inp = (False, False)
    assert trafos.OnBothSides(trafo)(*inp) == (True, True)
    assert trafos.OnInput(trafo)(*inp) == (True, False)
    assert trafos.OnTarget(trafo)(*inp) == (False, True)