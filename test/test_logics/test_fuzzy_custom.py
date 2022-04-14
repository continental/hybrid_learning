"""Tests for custom fuzzy logic operations."""

#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Dict, Any, Tuple, List, Type, Union

import numpy as np
import pytest
import torch

from hybrid_learning.fuzzy_logic import \
    product, lukasiewicz, goedel, boolean, Merge, FormulaParser
from hybrid_learning.fuzzy_logic.predicates import custom_ops

AND_ = boolean.AND
OR_ = boolean.OR


class TestIsPartOfA:
    """Test the custom predicate ``IsPartOfA``."""

    @pytest.mark.parametrize('setts', [
        dict(kernel_size=20),
        dict(thresh=.5),
        dict(conv_hang_front=False),
        dict(logical_and=product.AND),
        dict(logical_and=lukasiewicz.AND),
        dict(logical_and=goedel.AND),
        dict(logical_and=boolean.AND),
        dict(kernel_size=20, thresh=.5, conv_hang_front=False, logical_and=product.AND.variadic_()),
        dict(kernel_size=11, thresh=.5, logical_and=product.AND.variadic_()),
        dict(kernel_size=19, thresh=.1, conv_hang_front=True, logical_and=lukasiewicz.AND.variadic_()),
    ])
    def test_init(self, setts: Dict[str, Any]):
        """Check that the properties are calculated correctly."""
        # No logic given -> error
        with pytest.raises(TypeError):
            custom_ops.IsPartOfA("in_key", **{k: v for k, v in setts.items() if k != "logical_and"})
        setts.setdefault("logical_and", lukasiewicz.AND.variadic_())

        # direct init
        p = custom_ops.IsPartOfA("in_key", **setts)
        p_setts = p.settings
        for k, v in setts.items():
            assert p_setts[k] == v, "Wrong value for settings key {}".format(k)
        assert p.conv_hang_front == setts.get('conv_hang_front', True)

        # init with builder
        P = custom_ops.IsPartOfA.with_(**setts)
        p2 = P("in_key")
        assert isinstance(p2, custom_ops.IsPartOfA)
        p2_setts = p2.settings
        for k, v in setts.items():
            assert p2_setts[k] == v, "Wrong value for settings key {}".format(k)
        assert p2.conv_hang_front == setts.get('conv_hang_front', True)

        with pytest.raises(ValueError):
            custom_ops.IsPartOfA("in_key", **{**setts, 'kernel_size': -1})
        with pytest.raises(TypeError):
            custom_ops.IsPartOfA("in_key", **{**setts, 'kernel_size': .5})
        with pytest.raises(ValueError):
            custom_ops.IsPartOfA("in_key", **{**setts, 'thresh': -.5})
        with pytest.raises(ValueError):
            custom_ops.IsPartOfA("in_key", **{**setts, 'thresh': 1.5})

    @pytest.mark.parametrize('hang_front,kernel_size,thresh,center', [
        (True, 3, .5, (1, 1)),
        (True, 1, .5, (0, 0)),
        (True, 4, .5, (2, 2)), (False, 4, .5, (1, 1)),
        (True, 2, .75, (1, 1)), (False, 2, .75, (0, 0)),
        (True, 10, .1, (5, 5)), (False, 10, .1, (4, 4)),
    ])
    def test_create_ispartof_kernel(self, hang_front: bool, kernel_size: int,
                                    thresh: float, center: Tuple[int, int]):
        """Test kernel creation."""
        values = custom_ops.IsPartOfA.create_ispartof_kernel(kernel_size, thresh, conv_hang_front=not hang_front)

        # Correct type & shape
        assert isinstance(values, torch.Tensor)
        assert list(values.size()) == [1, 1, kernel_size, kernel_size, 1, 1]
        values = values.view(kernel_size, kernel_size)

        # Correct values
        assert values[center] == 1, "Wrong value at center_point {}".format(center)
        if kernel_size % 2 == 1:
            border_points = ((center[0], kernel_size - 1), (kernel_size - 1, center[1]),
                             (center[0], 0), (0, center[1]))
        elif hang_front:
            border_points = ((center[0], kernel_size - 1), (kernel_size - 1, center[1]),
                             (center[0], 1), (1, center[1]))
        else:
            border_points = ((center[0], kernel_size - 2), (kernel_size - 2, center[1]),
                             (center[0], 0), (0, center[1]))
        for border_point in border_points:
            if list(border_point) != list(center):
                assert np.allclose(values[border_point].item(), thresh), \
                    "Wrong value for border_point {}".format(border_point)

    @pytest.mark.parametrize('hang_front,kernel_size,thresh,center,mask', [
        # Boolean
        (True, 3, .5, (1, 1), [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        (False, 3, .5, (1, 1), [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        # Binary
        (True, 3, .5, (1, 1), [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        (True, 4, .2, (1, 1), [[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        (False, 5, .1, (1, 1), [[0, 0, 0], [0, 1, 0], [0, 1, 1]]),
        # Fuzzy
        (True, 3, .5, (1, 1), [[0, 0.5, 0], [0, 1, .25], [0, .4, .5]]),
        (False, 4, .6, (1, 1), [[1, 0, 0], [0, 0, 0], [0, 0, .2]]),
        (True, 5, .1, (1, 1), [[0, 0, 0], [.1, 1, .5], [0, .2, .1]]),
        # Non-square mask
        (True, 3, .5, (1, 1), [[0, 0, 0], [0, 0.5, 0], [0, 1, .25], [0, .4, .5]]),
        (False, 4, .6, (1, 1), [[1, 0, 0], [0, 0, .2]]),
        (True, 5, .1, (1, 1), [[0, 0, 0, .2], [0, .1, 1, .5], [0, 0, .2, .1]]),
    ])
    def test_operation(self, hang_front: bool, kernel_size: int,
                       thresh: float, center: Tuple[int, int], mask: List[List[float]]):
        """Test kernel creation."""
        inp: torch.Tensor = torch.tensor(mask).float()
        inp = inp.view(1, 1, *inp.size())
        logics = [lukasiewicz.Logic(), product.Logic(), goedel.Logic()]
        if (inp.bool() == inp).all():  # only check boolean.Logic for binary masks
            logics.append(boolean.Logic())
        for logic in logics:
            p = custom_ops.IsPartOfA("in_key", logical_and=lukasiewicz.AND.variadic_(),
                                     conv_hang_front=hang_front, kernel_size=kernel_size, thresh=thresh)
            outp: torch.Tensor = p({p.in_keys[0]: inp})[p.out_key]
            assert isinstance(outp, torch.Tensor)
            assert list(outp.size()) == list(inp.size())

            # New values must be greater or equal
            assert torch.greater_equal(outp, inp).all(), \
                "Input: {}\nOutput: {}".format(inp.tolist(), outp.tolist())

    @pytest.mark.parametrize('formula,parsed', [
        ('IsPartOfAa', custom_ops.IsPartOfA('a', logical_and=AND_.variadic_())),
        ('a||IsPartOfAb', OR_('a', custom_ops.IsPartOfA('b', logical_and=AND_.variadic_())))
    ])
    def test_parse(self, formula: str, parsed: Merge):
        logic = [*boolean.Logic(), custom_ops.IsPartOfA.with_(logical_and=AND_.variadic_())]
        parser = FormulaParser(logic)
        assert parser.parse(formula) == parsed


class CommonTestsAbstractFuzzyIntersect:
    """Common tests for classes derived from AbstractFuzzyIntersect."""
    LOGICAL_AND = product.AND.variadic_()
    LOGICAL_OR = goedel.OR.variadic_()
    OP = None

    def test_init(self):
        # Default:
        self.OP('a', 'b')  # default
        self.OP('a', 'b', logical_and=None, logical_or=None, mask_dims=(-2, -1), keep_dims=False)  # default

        # Standard calls that should work:
        self.OP('a', 'b', logical_and=self.LOGICAL_AND, logical_or=self.LOGICAL_OR)  # change logical ops
        self.OP('a', 'b', mask_dims=(-1, -2), keep_dims=True)  # change dim specs
        self.OP('a', 'b', logical_and=self.LOGICAL_AND, logical_or=self.LOGICAL_OR,
                mask_dims=(-1, -2), keep_dims=True)  # change all
        self.OP('a', 'b', mask_dims=-2)  # int mask_dims

        # Stuff that should fail:
        with pytest.raises(TypeError):
            self.OP('a', 'b', mask_dims=3.2)  # wrong mask_dims type
        with pytest.raises(TypeError):
            self.OP('a', 'b', mask_dims=[3.2])  # wrong mask_dims type
        with pytest.raises(TypeError):
            self.OP('a')  # wrong arity

    def _test_op_without_keep_dims(
        self, op_builder: Type[custom_ops.AbstractFuzzyIntersect],
        inp: Dict[str, torch.Tensor], expected: Union[float, np.ndarray], **init_args):
        expected_t: torch.FloatTensor = torch.tensor(expected).to(torch.float)
        op: custom_ops.AbstractFuzzyIntersect = op_builder('a', 'b', keep_dims=False, **init_args)

        squeezed_outp: torch.FloatTensor = op(inp)[op.out_key]
        assert isinstance(squeezed_outp, torch.Tensor)
        assert squeezed_outp.dtype == torch.float
        assert len(squeezed_outp.shape) == (len(inp['a'].shape) - len(op.mask_dims))
        assert torch.allclose(squeezed_outp, expected_t), \
            "Wrong result for keep_dims=False:\n Expected\t{}\n Got\t{}".format(expected_t, squeezed_outp)

    def _test_op_with_keep_dims(
        self, op_builder: Type[custom_ops.AbstractFuzzyIntersect],
        inp: Dict[str, torch.Tensor], expected: Union[float, np.ndarray], **init_args):
        expected_t: torch.FloatTensor = torch.tensor(expected).to(torch.float)
        op: custom_ops.AbstractFuzzyIntersect = op_builder('a', 'b', keep_dims=True, **init_args)

        assert op.keep_dims
        outp: torch.FloatTensor = op(inp)[op.out_key]
        assert isinstance(outp, torch.Tensor)
        assert outp.dtype == torch.float
        assert len(outp.shape) == len(inp['a'].shape)
        abs_mask_dims = [d if d>=0 else len(inp['a'].shape)+d for d in op.mask_dims]
        assert all(outp.shape[i] == inp['a'].shape[i] for i in range(len(inp['a'].shape))
                   if i not in abs_mask_dims), \
            "Wrong results shape for keep_dims=True, mask_dims={} (resp. {}):\n mask_a.shape:\t{}\n gotten shape:\t{}"\
                .format(op.mask_dims, abs_mask_dims, inp['a'].shape, outp.shape)
        assert all(outp.shape[i] == 1 for i in op.mask_dims)
        assert torch.allclose(outp, expected_t.view(outp.shape)), \
            "Wrong result for keep_dims=True:\n Expected\t{}\n Got\t{}".format(expected_t, outp)
        



class TestCoveredBy(CommonTestsAbstractFuzzyIntersect):
    OP = custom_ops.CoveredBy

    @pytest.mark.parametrize('mask_a,mask_b,expected,further_init_args', [
        # 1x1 binary mask
        (np.array([[1]]), np.array([[1]]), 1, {}),
        (np.array([[1]]), np.array([[0]]), 0, {}),
        (np.array([[0]]), np.array([[1]]), 1, {}),
        (np.array([[0]]), np.array([[0]]), 1, {}),
        # 2x2 binary all-same mask
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), 1, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), 0, {}),
        (np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), 1, {}),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1, {}),
        # More dims
        (np.array([[[[1, 1]], [[1, 0]]]]), np.array([[[[1, 0]], [[1, 1]]]]), 2/3, dict(mask_dims=[-3,-1])),
        # 2x2 binary masks, a all same
        (np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 0]]), 0.25, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [0, 0]]), 0.5, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 0]]), 0.75, {}),
        # 2x2 binary masks
        (np.array([[1, 1], [1, 0]]), np.array([[1, 1], [1, 0]]), 1, {}),
        (np.array([[1, 1], [0, 1]]), np.array([[1, 1], [1, 0]]), 2 / 3, {}),
        (np.array([[1, 1], [0, 1]]), np.array([[0, 1], [1, 0]]), 1 / 3, {}),
        # 2x2 binary masks (bool values)
        (np.array([[True, True], [True, False]]), np.array([[True, True], [True, False]]), 1., {}),
        (np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]]), 2 / 3, {}),
        (np.array([[True, True], [False, True]]), np.array([[False, True], [True, False]]), 1 / 3, {}),
        # 2x2 non-binary masks
        (np.array([[1, .5], [1, 0]]), np.array([[.25, .5], [.5, 0]]), 1 / 2.5, {}),
        (np.array([[.25, .5], [.5, 0]]), np.array([[1, .5], [1, 0]]), 1 / 1.25, {}),  # check assymmetry
        (np.array([[1, .25], [0, .75]]), np.array([[.25, 1], [1, 0]]), .5 / 2, {}),
        (np.array([[.1, .2], [0, .7]]), np.array([[0, .5], [1, 0]]), .1 / 1, {}),
        # 2x2 mixed binary and non-binar
        (np.array([[True, True], [True, False]]), np.array([[.25, .5], [.5, 0]]), 1.25 / 3, {}),
        (np.array([[1, .25], [0, .75]]), np.array([[False, True], [True, False]]), .25 / 2, {}),
    ])
    def test_op(self, mask_a: np.ndarray, mask_b: np.ndarray, expected: Union[float, np.ndarray], further_init_args: Dict[str, Any]):
        self._test_op_without_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)
        self._test_op_with_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)
        

class TestIoUWith(CommonTestsAbstractFuzzyIntersect):
    OP = custom_ops.IoUWith

    @pytest.mark.parametrize('mask_a,mask_b,expected,further_init_args', [
        # 1x1 binary mask
        (np.array([[1]]), np.array([[1]]), 1, {}),
        (np.array([[1]]), np.array([[0]]), 0, {}),
        (np.array([[0]]), np.array([[1]]), 0, {}),
        (np.array([[0]]), np.array([[0]]), 1, {}),
        # 2x2 binary all-same mask
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), 1, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), 0, {}),
        (np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), 0, {}),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1, {}),
        # More dims
        (np.array([[[[1, 1]], [[1, 0]]]]), np.array([[[[1, 0]], [[1, 1]]]]), 2/4, dict(mask_dims=[-3,-1])),
        # 2x2 binary masks, a all one
        (np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 0]]), 0.25, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [0, 0]]), 0.5, {}),
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 0]]), 0.75, {}),
        # 2x2 binary masks
        (np.array([[1, 1], [1, 0]]), np.array([[1, 1], [1, 0]]), 1, {}),
        (np.array([[1, 1], [0, 1]]), np.array([[1, 1], [1, 0]]), 2 / 4, {}),
        (np.array([[1, 1], [0, 1]]), np.array([[0, 1], [1, 0]]), 1 / 4, {}),
        # 2x2 binary masks (bool values)
        (np.array([[True, True], [True, False]]), np.array([[True, True], [True, False]]), 1., {}),
        (np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]]), 2 / 4, {}),
        (np.array([[True, True], [False, True]]), np.array([[False, True], [True, False]]), 1 / 4, {}),
        # 2x2 non-binary masks with default AND and OR
        (np.array([[1, .5], [1, 0]]), np.array([[.25, .5], [.5, 0]]), (.25+.25+.5) / (1+.75+1), {}),
        (np.array([[1, .2], [0, .75]]), np.array([[.25, .5], [1, 0]]), (.25+.1) / (1+.6+1+.75), {}),
        (np.array([[.1, .2], [0, .7]]), np.array([[0, .5], [1, 0]]), .1 / (.1+.6+1+.7), {}),
        # 2x2 non-binary masks with goedel AND
        (np.array([[1, .5], [1, 0]]), np.array([[.25, .5], [.5, 0]]), (.25+.5+.5) / (1+.5+1), dict(logical_and=goedel.AND.variadic_())),
        (np.array([[1, .25], [0, .75]]), np.array([[.25, 1], [1, 0]]), (.25+.25) / (1+1+1+.75), dict(logical_and=goedel.AND.variadic_())),
        (np.array([[1, .2], [0, .75]]), np.array([[.25, .5], [1, 0]]), (.25+.2) / (1+.5+1+.75), dict(logical_and=goedel.AND.variadic_())),
        (np.array([[.1, .2], [0, .7]]), np.array([[0, .5], [1, 0]]), .2 / (.1+.5+1+.7), dict(logical_and=goedel.AND.variadic_())),
        # 2x2 mixed binary and non-binary
        (np.array([[True, True], [True, False]]), np.array([[.25, .5], [.5, 0]]), 1.25 / 3, {}),
        (np.array([[1, .25], [0, .75]]), np.array([[False, True], [True, False]]), .25 / (1+1+1+.75), {}),
    ])
    def test_op(self, mask_a: np.ndarray, mask_b: np.ndarray, expected: Union[float, np.ndarray], further_init_args: Dict[str, Any]):
        self._test_op_without_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)
        self._test_op_with_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)
        
        # Check symmetry:
        self._test_op_without_keep_dims(self.OP, {'b': torch.tensor(mask_a), 'a': torch.tensor(mask_b)}, expected, **further_init_args)
        self._test_op_with_keep_dims(self.OP, {'b': torch.tensor(mask_a), 'a': torch.tensor(mask_b)}, expected, **further_init_args)


class TestBestIoUWith(TestIoUWith):
    OP = custom_ops.BestIoUWith
    LOGICAL_AND = product.AND.variadic_()

    @pytest.mark.parametrize('mask_a,mask_b,expected,further_init_args', [
        # 1x1 binary mask
        (np.array([[1]]), np.array([[0], [1]]), 1, dict(mask_dims=-1)),
        (np.array([[0]]), np.array([[0], [1]]), 1, dict(mask_dims=-1)),
        (np.array([[0]]), np.array([[0.5], [1]]), 0, dict(mask_dims=-1)),
        (np.array([[0], [1]]), np.array([[1]]), [0, 1], dict(mask_dims=-1)),
        (np.array([[0], [1]]), np.array([[0]]), [1, 0], dict(mask_dims=-1)),
        (np.array([[.5], [1]]), np.array([[0]]), [0, 0], dict(mask_dims=-1)),
        (np.array([[1], [0]]), np.array([[0], [1]]), [1, 1], dict(mask_dims=-1)),
        (np.array([[1], [1]]), np.array([[0], [0]]), [0, 0], dict(mask_dims=-1)),
        (np.array([[0], [1]]), np.array([[1], [1]]), [0, 1], dict(mask_dims=-1)),
    ])
    def test_op_with_more_dims(self, mask_a: np.ndarray, mask_b: np.ndarray, expected: Union[float, np.ndarray], further_init_args: Dict[str, Any]):
        self._test_op_without_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)
        self._test_op_with_keep_dims(self.OP, {'a': torch.tensor(mask_a), 'b': torch.tensor(mask_b)}, expected, **further_init_args)