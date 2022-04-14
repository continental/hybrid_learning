"""Tests for the different pre-defined fuzzy logics."""
#  Copyright (c) 2022 Continental Automotive GmbH

from collections import OrderedDict
from typing import Dict, Any, Union, Callable

import numpy as np
import pyparsing
import pytest
import torch

from hybrid_learning.fuzzy_logic import logic_by_name
from hybrid_learning.fuzzy_logic import \
    Logic, lukasiewicz, Merge, boolean, goedel, product, FormulaParser
from hybrid_learning.fuzzy_logic.tnorm_connectives import fuzzy_common
from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import BoolTorchOrNumpyOperation


def common_test_parsing(formula: str, parsed: Merge, parse: FormulaParser):
    """Test main properties of parsing like idempotence"""
    # wrong formulas
    if parsed is None:
        with pytest.raises(pyparsing.ParseException):
            parse(formula)
        return
    parse_out = parse(formula)
    # parsing and usual calling interchangeable
    assert parse_out == parsed
    # idempotence
    assert parse_out.to_str(use_whitespace=False, precedence=parse.logic) == formula
    assert parse.to_str(formula, use_whitespace=False) == formula
    # out_key option
    assert parse(formula, out_key="c").out_key == "c"
    assert not parse(formula, overwrite=False).overwrite


def common_test_call(operation: Merge, annotations: Dict[str, Any],
                     expected: Union[bool, np.ndarray], primitive: Callable = None):
    """Common test routine for calling different fuzzy logic operators."""
    # Default
    changed_anns = dict(annotations)
    out_anns = operation(changed_anns)
    assert out_anns is changed_anns
    assert operation.out_key in out_anns
    outp = out_anns[operation.out_key]
    assert np.allclose(outp - expected, 0), \
        "Given: {}\nExpected: {}".format(outp.tolist(), expected.tolist())
    if primitive is not None:
        primitive_outp = primitive([annotations[k] for k in operation.operation_keys])
        assert np.allclose(outp, primitive_outp)

    # All torch tensors
    changed_anns = {k: torch.as_tensor(annotations[k]) for k in operation.operation_keys}
    outp = operation(changed_anns)[operation.out_key]
    assert isinstance(outp, torch.Tensor)
    assert outp.dtype == list(changed_anns.values())[0].dtype
    assert torch.allclose(outp, torch.as_tensor(expected, dtype=outp.dtype)), \
        "Given: {}\nExpected: {}".format(outp.tolist(), expected.tolist())
    if primitive is not None:
        primitive_outp = primitive([changed_anns[k] for k in operation.operation_keys])
        assert torch.allclose(outp, primitive_outp)

    # Some torch tensors
    changed_anns = {**{k: annotations[k] for k in sorted(operation.operation_keys)[:-1]},
                    **{k: torch.as_tensor(annotations[k]) for k in [sorted(operation.operation_keys)[-1]]}}
    outp = operation(changed_anns)[operation.out_key]
    assert isinstance(outp, torch.Tensor)
    assert torch.allclose(outp, torch.as_tensor(expected, dtype=outp.dtype))
    if primitive is not None:
        primitive_outp = primitive([changed_anns[k] for k in operation.operation_keys])
        assert torch.allclose(outp, primitive_outp)


class TestLukasiewicz:
    """Test Lukasiewicz logical operations."""
    LOGIC: Logic = lukasiewicz.LukasiewiczLogic()

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (lukasiewicz.AND('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (lukasiewicz.AND('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (lukasiewicz.AND('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (lukasiewicz.AND('a', 'b'), dict(a=np.ones([2, 2]), b=np.array([0, .5])), np.array([[0, .5], [0, .5]])),
        # Fuzzy
        (lukasiewicz.AND('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)), np.ones(3) * .25),
        # Ignore one argument
        (lukasiewicz.AND('a', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)), np.ones(3) * .5),
    ])
    def test_and(self, operation: Merge, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('AND'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (lukasiewicz.OR('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (lukasiewicz.OR('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (lukasiewicz.OR('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (lukasiewicz.OR('a', 'b'), dict(a=np.ones([2, 2]) * .25, b=np.array([0, .5])),
         np.array([[.25, .75], [.25, .75]])),
        # Non-Boolean
        (lukasiewicz.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(3)),
        # Ignore one argument
        (lukasiewicz.OR('a', 'b'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(3) * .75),
    ])
    def test_or(self, operation: Merge, annotations: Dict[str, Any],
                expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('OR'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # Boolean
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (lukasiewicz.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3), B=np.zeros(3)), np.zeros(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3)), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.zeros(3), b=np.zeros(3)), np.ones(3)),
        # Test broadcasting
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.ones([2, 2]), b=np.array([0, .75])),
         np.array([[0, .75], [0, .75]])),
        # "Fuzzy"
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .75, b=np.ones(3) * .5), np.ones(3) * .75),
        (lukasiewicz.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .75, B=np.ones(3) * .25), np.ones(3) * .5),
        (lukasiewicz.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .5, B=np.ones(3) * .25), np.ones(3) * .75),
        # First argument constant 1
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .1), np.ones(3) * .1),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .3), np.ones(3) * .3),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .6), np.ones(3) * .6),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .9), np.ones(3) * .9),
        # First argument constant 0
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .1), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .3), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .6), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .9), np.ones(3)),
        # Second argument constant 1
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3)), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.ones(3)), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.ones(3)), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.ones(3)), np.ones(3)),
        # Second argument constant 0
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.zeros(3)), np.ones(3) * .9),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.zeros(3)), np.ones(3) * .7),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.zeros(3)), np.ones(3) * .4),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.zeros(3)), np.ones(3) * .1),
        # Tautology
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .1), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .3), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .5), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .6), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .9), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'a'), OrderedDict(a=np.zeros(3)), np.ones(3)),
        # Ignore one argument
        (lukasiewicz.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)),
         np.ones(3)),
        # arguments very close
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3) * .1 - 1e-08), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * 1e-08), np.ones(3)),
        (lukasiewicz.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * 1e-08, c=np.zeros(3)), np.ones(3)),
    ])
    def test_implies(self, operation: Merge, annotations: Dict[str, Any],
                     expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('>>'))
        inverse_op = lukasiewicz.IMPLIEDBY(*reversed(operation.in_keys))
        common_test_call(operation=inverse_op, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('<<'))

        # Lukasiewicz implies is equivalent to OR(NOT("a"), "b"):
        not_a_or_b = lukasiewicz.OR(lukasiewicz.NOT(operation.in_keys[0]), operation.in_keys[1])
        op_out, not_a_or_b_out = operation(annotations), not_a_or_b(annotations)
        for key, val in not_a_or_b_out.items():
            assert key in op_out
            assert np.allclose(op_out[key], val)

    @pytest.mark.parametrize('formula,parsed', [
        ("a&&b", lukasiewicz.AND('a', 'b')),
        ("a||b", lukasiewicz.OR('a', 'b')),
        ("~c", fuzzy_common.NOT('c')),
        ("a&&b||c", lukasiewicz.AND('a', lukasiewicz.OR('b', 'c'))),
        ("a&&b||~c", lukasiewicz.AND('a', lukasiewicz.OR('b', fuzzy_common.NOT('c')))),
        ("&&||", None), ("~", None),
    ])
    def test_parsing(self, formula, parsed):
        """Test main properties of parsing like idempotence"""
        common_test_parsing(formula=formula, parsed=parsed, parse=self.LOGIC.parser())

    @pytest.mark.parametrize('is_pure,formula', [
        (True, lukasiewicz.AND(lukasiewicz.OR('a', 'b'), 'c', lukasiewicz.NOT('d'))),
        (False, boolean.AND(lukasiewicz.OR('a', 'b'), boolean.NOT('c'), 'd')),
    ])
    def test_is_pure(self, formula: Merge, is_pure: bool):
        """Test is_pure."""
        assert self.LOGIC.is_pure(formula) == is_pure


class TestGoedel:
    """Test Goedel logical operations."""
    LOGIC: Logic = goedel.GoedelLogic()

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (goedel.AND('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (goedel.AND('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (goedel.AND('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (goedel.AND('a', 'b'), dict(a=np.ones([2, 2]), b=np.array([0, .5])), np.array([[0, .5], [0, .5]])),
        # Fuzzy
        (goedel.AND('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)), np.ones(3) * .5),
        # Ignore one argument
        (goedel.AND('a', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)), np.ones(3) * .5),
    ])
    def test_and(self, operation: Merge, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('AND'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (goedel.OR('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (goedel.OR('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (goedel.OR('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (goedel.OR('a', 'b'), dict(a=np.ones([2, 2]) * .25, b=np.array([0, .5])), np.array([[.25, .5], [.25, .5]])),
        # Non-Boolean
        (goedel.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(1)),
        # Ignore one argument
        (goedel.OR('a', 'b'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(3) * .5),
    ])
    def test_or(self, operation: Merge, annotations: Dict[str, Any],
                expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('OR'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # Boolean
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (goedel.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3), B=np.zeros(3)), np.zeros(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3)), np.ones(3)),
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.zeros(3), b=np.zeros(3)), np.ones(3)),
        # Test broadcasting
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.ones([2, 2]), b=np.array([0, .75])),
         np.array([[0, .75], [0, .75]])),
        # "Fuzzy"
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75), np.ones(3)),
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .75, b=np.ones(3) * .5), np.ones(3) * .5),
        (goedel.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .75, B=np.ones(3) * .25), np.ones(3) * .25),
        (goedel.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .5, B=np.ones(3) * .25), np.ones(3) * .25),
        # First argument constant 1
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .1), np.ones(3) * .1),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .3), np.ones(3) * .3),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .6), np.ones(3) * .6),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .9), np.ones(3) * .9),
        # First argument constant 0
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .1), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .3), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .6), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .9), np.ones(3)),
        # Second argument constant 1
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3)), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.ones(3)), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.ones(3)), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.ones(3)), np.ones(3)),
        # Second argument constant 0
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.zeros(3)), np.zeros(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.zeros(3)), np.zeros(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.zeros(3)), np.zeros(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.zeros(3)), np.zeros(3)),
        # Tautology
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .1), np.ones(3)),
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .3), np.ones(3)),
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .5), np.ones(3)),
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .6), np.ones(3)),
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .9), np.ones(3)),
        (goedel.IMPLIES('a', 'a'), OrderedDict(a=np.zeros(3)), np.ones(3)),
        # Ignore one argument
        (goedel.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)),
         np.ones(3)),
        # arguments very close
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3) * .1 - 1e-08), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * 1e-08), np.ones(3)),
        (goedel.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * 1e-08, c=np.zeros(3)), np.ones(3)),
    ])
    def test_implies(self, operation: Merge, annotations: Dict[str, Any],
                     expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('>>'))
        inverse_op = goedel.IMPLIEDBY(*reversed(operation.in_keys))
        common_test_call(operation=inverse_op, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('<<'))

    @pytest.mark.parametrize('formula,parsed', [
        ("a&&b", goedel.AND('a', 'b')),
        ("a||b", goedel.OR('a', 'b')),
        ("~c", fuzzy_common.NOT('c')),
        ("a&&b||c", goedel.AND('a', goedel.OR('b', 'c'))),
        ("a&&b||~c", goedel.AND('a', goedel.OR('b', fuzzy_common.NOT('c')))),
        ("&&||", None), ("~", None),
    ])
    def test_parsing(self, formula, parsed):
        """Test main properties of parsing like idempotence"""
        common_test_parsing(formula=formula, parsed=parsed, parse=self.LOGIC.parser())

    @pytest.mark.parametrize('is_pure,formula', [
        (True, goedel.AND(goedel.OR('a', 'b'), 'c', goedel.NOT('d'))),
        (False, boolean.AND(goedel.OR('a', 'b'), boolean.NOT('c'), 'd')),
        (False, lukasiewicz.AND(goedel.OR('a', 'b'), boolean.NOT('c'), 'd')),
    ])
    def test_is_pure(self, formula: Merge, is_pure: bool):
        """Test is_pure."""
        assert self.LOGIC.is_pure(formula) == is_pure


class TestProduct:
    """Test Product logical operations."""
    LOGIC: Logic = product.ProductLogic()

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (product.AND('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (product.AND('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (product.AND('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (product.AND('a', 'b'), dict(a=np.ones([2, 2]) * .5, b=np.array([0, .5])), np.array([[0, .25], [0, .25]])),
        # Fuzzy
        (product.AND('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .8, c=np.ones(3)), np.ones(3) * .4),
        # Ignore one argument
        (product.AND('a', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .8, c=np.ones(3) * .6), np.ones(3) * .3),
    ])
    def test_and(self, operation: Merge, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('AND'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # 1-argument = identity
        (product.OR('a'), dict(a=np.ones(4) * 0.3), np.ones(4) * 0.3),
        # Boolean
        (product.OR('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (product.OR('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3)),
        # Test broadcasting
        (product.OR('a', 'b'), dict(a=np.ones([2, 2]) * .5, b=np.array([0, .5])), np.array([[.5, .75], [.5, .75]])),
        # Non-Boolean
        (product.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(1)),
        (product.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3) * .8), np.ones(1) * .925),
        # Ignore one argument
        (product.OR('a', 'b'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(3) * .625),
    ])
    def test_or(self, operation: Merge, annotations: Dict[str, Any],
                expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('OR'))

    @pytest.mark.parametrize('operation,annotations,expected', [
        # Boolean
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3), b=np.ones(3)), np.ones(3)),
        (product.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3), B=np.zeros(3)), np.zeros(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3)), np.ones(3)),
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.zeros(3), b=np.zeros(3)), np.ones(3)),
        # Test broadcasting
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.ones([2, 2]), b=np.array([0, .75])),
         np.array([[0, .75], [0, .75]])),
        # "Fuzzy"
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75), np.ones(3)),
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .75, b=np.ones(3) * .5), np.ones(3) * 2 / 3),
        (product.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .75, B=np.ones(3) * .25), np.ones(3) / 3),
        (product.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .5, B=np.ones(3) * .25), np.ones(3) / 2),
        # First argument constant 1
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .1), np.ones(3) * .1),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .3), np.ones(3) * .3),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .6), np.ones(3) * .6),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3), c=np.ones(3) * .9), np.ones(3) * .9),
        # First argument constant 0
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .1), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .3), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .6), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * .9), np.ones(3)),
        # Second argument constant 1
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3)), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.ones(3)), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.ones(3)), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.ones(3)), np.ones(3)),
        # Second argument constant 0
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.zeros(3)), np.zeros(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .3, c=np.zeros(3)), np.zeros(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .6, c=np.zeros(3)), np.zeros(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .9, c=np.zeros(3)), np.zeros(3)),
        # Tautology
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .1), np.ones(3)),
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .3), np.ones(3)),
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .5), np.ones(3)),
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .6), np.ones(3)),
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .9), np.ones(3)),
        (product.IMPLIES('a', 'a'), OrderedDict(a=np.zeros(3)), np.ones(3)),
        # Ignore one argument
        (product.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)),
         np.ones(3)),
        # arguments very close
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3) * .1 - 1e-08), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3) * 1e-08), np.ones(3)),
        (product.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * 1e-08, c=np.zeros(3)), np.ones(3)),
    ])
    def test_implies(self, operation: Merge, annotations: Dict[str, Any],
                     expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('>>'))
        inverse_op = product.IMPLIEDBY(*reversed(operation.in_keys))
        common_test_call(operation=inverse_op, annotations=annotations, expected=expected,
                         primitive=self.LOGIC.logical_('<<'))


    @pytest.mark.parametrize('formula,parsed', [
        ("a&&b", product.AND('a', 'b')),
        ("a||b", product.OR('a', 'b')),
        ("~c", fuzzy_common.NOT('c')),
        ("a&&b||c", product.AND('a', product.OR('b', 'c'))),
        ("a&&b||~c", product.AND('a', product.OR('b', fuzzy_common.NOT('c')))),
        ("&&||", None), ("~", None),
    ])
    def test_parsing(self, formula, parsed):
        """Test main properties of parsing like idempotence"""
        common_test_parsing(formula=formula, parsed=parsed, parse=self.LOGIC.parser())

    @pytest.mark.parametrize('is_pure,formula', [
        (True, product.AND(product.OR('a', 'b'), 'c', product.NOT('d'))),
        (False, boolean.AND(product.OR('a', 'b'), boolean.NOT('c'), 'd')),
        (False, lukasiewicz.AND(product.OR('a', 'b'), boolean.NOT('c'), 'd')),
        (False, product.AND(goedel.OR('a', 'b'), boolean.NOT('c'), 'd')),
    ])
    def test_is_pure(self, formula: Merge, is_pure: bool):
        """Test is_pure."""
        assert self.LOGIC.is_pure(formula) == is_pure


class TestBoolean:
    """Test Boolean logical operations."""
    LOGIC: Logic = boolean.BooleanLogic()

    @pytest.mark.parametrize('operation,annotations,expected,bool_thresh', [
        # 1-argument = identity
        (boolean.AND('a'), dict(a=np.ones(4) * 0.3), np.zeros(4), 0.5),
        (boolean.AND('a'), dict(a=np.ones(4) * 0.6), np.ones(4), 0.5),
        # Boolean
        (boolean.AND('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3), 0.5),
        (boolean.AND('a', 'b'), dict(a=np.ones(3), b=np.zeros(3)), np.zeros(3), 0.5),
        (boolean.AND('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3), 0.5),
        # Test broadcasting
        (boolean.AND('a', 'b'), dict(a=np.ones([2, 2]) * .75, b=np.array([0, .75])), np.array([[0, 1], [0, 1]]), 0.5),
        # "Fuzzy"
        (boolean.AND('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .8, c=np.ones(3)), np.ones(3), 0.5),
        (boolean.AND('a', 'b', 'c'), dict(a=np.ones(3) * .49, b=np.ones(3) * .8, c=np.ones(3)), np.zeros(3), 0.5),
        # Different bool_thresh
        (boolean.AND('a', 'b'), dict(a=np.ones(3) * .4, b=np.ones(3) * .8), np.ones(3), 0.4),
        (boolean.AND('a', 'b'), dict(a=np.ones(3) * .4, b=np.ones(3) * .8), np.zeros(3), 0.5),
        (boolean.AND('a', 'b'), dict(a=np.ones(3) * .5, b=np.ones(3) * .8), np.ones(3), 0.5),
        (boolean.AND('a', 'b'), dict(a=np.ones(3) * .5, b=np.ones(3) * .8), np.zeros(3), 0.6),
        (boolean.AND('a'), dict(a=np.ones(3) * .8), np.zeros(3), 0.9),
        # Ignore one argument
        (boolean.AND('a', 'c'), dict(a=np.ones(3) * .55, b=np.ones(3) * .3, c=np.ones(3) * .6), np.ones(3), 0.5),
    ])
    def test_and(self, operation: BoolTorchOrNumpyOperation, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray], bool_thresh: float):
        primitive: BoolTorchOrNumpyOperation = self.LOGIC.logical_('AND')
        operation.bool_thresh, primitive.bool_thresh = bool_thresh, bool_thresh
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=primitive)

    @pytest.mark.parametrize('operation,annotations,expected,bool_thresh', [
        # 1-argument = identity
        (boolean.OR('a'), dict(a=np.ones(4) * 0.3), np.zeros(4), 0.5),
        (boolean.OR('a'), dict(a=np.ones(4) * 0.6), np.ones(4), 0.5),
        # Boolean
        (boolean.OR('a', 'b'), dict(a=np.ones(3), b=np.ones(3)), np.ones(3), 0.5),
        (boolean.OR('a', 'b'), dict(a=np.ones(3), b=np.zeros(3)), np.ones(3), 0.5),
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), np.ones(3), 0.5),
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3), b=np.ones(3), c=np.zeros(3)), np.ones(3), 0.5),
        # Test broadcasting
        (boolean.OR('a', 'b'), dict(a=np.zeros([2, 2]), b=np.array([0, .75])), np.array([[0, 1], [0, 1]]), 0.5),
        # "Fuzzy"
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3)), np.ones(1), 0.5),
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3) * .8), np.ones(1), 0.5),
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3) * .5, b=np.ones(3) * .25, c=np.ones(3) * .3), np.ones(1), 0.5),
        (boolean.OR('a', 'b', 'c'), dict(a=np.ones(3) * .49, b=np.ones(3) * .25, c=np.ones(3) * .3), np.zeros(1), 0.5),
        # Different bool_threshs
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .2, b=np.ones(3) * .4), np.ones(1), 0.1),
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .2, b=np.ones(3) * .4), np.ones(1), 0.2),
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .2, b=np.ones(3) * .4), np.ones(1), 0.4),
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .2, b=np.ones(3) * .4), np.zeros(1), 0.5),
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .2, b=np.ones(3) * .4), np.zeros(1), 0.6),
        # Ignore one argument
        (boolean.OR('a', 'b'), dict(a=np.ones(3) * .4, b=np.ones(3) * .25, c=np.ones(3)), np.zeros(3), 0.5),
    ])
    def test_or(self, operation: BoolTorchOrNumpyOperation, annotations: Dict[str, Any],
                expected: Union[bool, np.ndarray], bool_thresh: float):
        primitive: BoolTorchOrNumpyOperation = self.LOGIC.logical_('OR')
        operation.bool_thresh, primitive.bool_thresh = bool_thresh, bool_thresh
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=primitive)

    @pytest.mark.parametrize('operation,annotations,expected,bool_thresh', [
        # Boolean
        (boolean.NOT('a'), dict(a=np.ones(4)), np.zeros(4), 0.5),
        (boolean.NOT('a'), dict(a=np.zeros(4)), np.ones(4), 0.5),
        # "Fuzzy"
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.3), np.ones(4), 0.5),
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.6), np.zeros(4), 0.5),
        (boolean.NOT('b'), dict(b=np.ones(4) * 0.5), np.zeros(4), 0.5),
        (boolean.NOT('b'), dict(b=np.ones(4) * 0.49), np.ones(4), 0.5),
        # Different bool_thresh
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.5), np.ones(4), 0.6),
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.3), np.zeros(4), 0.3),
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.3), np.zeros(4), 0.2),
        (boolean.NOT('a'), dict(a=np.ones(4) * 0.6), np.ones(4), 0.8),
    ])
    def test_not(self, operation: BoolTorchOrNumpyOperation, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray], bool_thresh: float):
        primitive: BoolTorchOrNumpyOperation = self.LOGIC.logical_('NOT')
        operation.bool_thresh, primitive.bool_thresh = bool_thresh, bool_thresh
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=primitive)

    @pytest.mark.parametrize('operation,annotations,expected,bool_thresh', [
        # Boolean
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3), b=np.ones(3)), np.ones(3), 0.5),
        (boolean.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3), B=np.zeros(3)), np.zeros(3), 0.5),
        (boolean.IMPLIES('a', 'c'), OrderedDict(a=np.zeros(3), c=np.ones(3)), np.ones(3), 0.5),
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.zeros(3), b=np.zeros(3)), np.ones(3), 0.5),
        # Test broadcasting
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones([2, 2]), b=np.array([0, .75])),
         np.array([[0, 1], [0, 1]]), 0.5),
        # "Fuzzy"
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .51, b=np.ones(3) * .75), np.ones(3), 0.5),
        (boolean.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .6, B=np.ones(3) * .5), np.ones(3), 0.5),
        (boolean.IMPLIES('A', 'B'), OrderedDict(A=np.ones(3) * .6, B=np.ones(3) * .49), np.zeros(3), 0.5),
        (boolean.IMPLIES('a', 'c'), OrderedDict(a=np.ones(3) * .1, c=np.ones(3) * .8), np.ones(3), 0.5),
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3), b=np.ones(3) * .9), np.ones(3), 0.5),
        # Different bool threshs
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .7, b=np.ones(3) * .51), np.zeros(3), 0.7),
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .7, b=np.ones(3) * .51), np.ones(3), 0.8),
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .7, b=np.ones(3) * .4), np.ones(3), 0.4),
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .7, b=np.ones(3) * .5), np.zeros(3), 0.6),
        # Ignore one argument
        (boolean.IMPLIES('a', 'b'), OrderedDict(a=np.ones(3) * .5, b=np.ones(3) * .75, c=np.ones(3)),
         np.ones(3), 0.5),
        # Tautology
        (boolean.IMPLIES('a', 'a'), OrderedDict(a=np.ones(3) * .5), np.ones(3), 0.5),
    ])
    def test_implies(self, operation: BoolTorchOrNumpyOperation, annotations: Dict[str, Any],
                     expected: Union[bool, np.ndarray], bool_thresh: float):
        primitive: BoolTorchOrNumpyOperation = self.LOGIC.logical_('>>')
        operation.bool_thresh, primitive.bool_thresh = bool_thresh, bool_thresh
        common_test_call(operation=operation, annotations=annotations, expected=expected,
                         primitive=primitive)
        inverse_op: BoolTorchOrNumpyOperation = boolean.IMPLIEDBY(*reversed(operation.in_keys))
        primitive: BoolTorchOrNumpyOperation = self.LOGIC.logical_('<<')
        inverse_op.bool_thresh, primitive.bool_thresh = bool_thresh, bool_thresh
        common_test_call(operation=inverse_op, annotations=annotations, expected=expected,
                         primitive=primitive)

    @pytest.mark.parametrize('formula,parsed', [
        ("a&&b", boolean.AND('a', 'b')),
        ("a||b", boolean.OR('a', 'b')),
        ("~c", boolean.NOT('c')),
        ("a&&b||c", boolean.AND('a', boolean.OR('b', 'c'))),
        ("a&&b||~c", boolean.AND('a', boolean.OR('b', boolean.NOT('c')))),
        ("&&||", None), ("~", None),
    ])
    def test_parsing(self, formula, parsed):
        """Test main properties of parsing like idempotence"""
        common_test_parsing(formula=formula, parsed=parsed, parse=self.LOGIC.parser())

    @pytest.mark.parametrize('is_pure,formula', [
        (True, boolean.AND(boolean.OR('a', 'b'), 'c', boolean.NOT('d'))),
        (False, boolean.AND(product.OR('a', 'b'), boolean.NOT('c'), 'd')),
        (False, lukasiewicz.AND(product.OR('a', 'b'), boolean.NOT('c'), 'd')),
        (False, product.AND(goedel.OR('a', 'b'), boolean.NOT('c'), 'd')),
    ])
    def test_is_pure(self, formula: Merge, is_pure: bool):
        """Test is_pure."""
        assert self.LOGIC.is_pure(formula) == is_pure


class TestCommon:
    """Test common fuzzy logic operations."""

    @pytest.mark.parametrize('operation,annotations,expected', [
        # Boolean
        (fuzzy_common.NOT('a'), dict(a=np.ones(4)), np.zeros(4)),
        (fuzzy_common.NOT('a'), dict(a=np.zeros(4)), np.ones(4)),
        (fuzzy_common.NOT('a'), dict(a=np.zeros(4), b=np.ones(4)), np.ones(4)),
        # Non-Boolean
        (fuzzy_common.NOT('b'), dict(b=np.ones(3) * .5), np.ones(3) * .5),
        (fuzzy_common.NOT('b'), dict(b=np.ones(3) * .75), np.ones(3) * .25),
        (fuzzy_common.NOT('b'), dict(b=np.ones(3) * .25), np.ones(3) * .75),
    ])
    def test_not(self, operation: Merge, annotations: Dict[str, Any],
                 expected: Union[bool, np.ndarray]):
        common_test_call(operation=operation, annotations=annotations, expected=expected)

    @pytest.mark.parametrize('key,logic_mod', [
        ('lukasiewicz', lukasiewicz),
        ('goedel', goedel),
        ('product', product),
        ('boolean', boolean),
    ])
    def test_logic_by_name(self, key: str, logic_mod):
        logic = logic_by_name(key)
        assert isinstance(logic, logic_mod.Logic)
        assert logic.op('AND') is logic_mod.AND
        assert logic.op('OR') is logic_mod.OR
        assert logic.op('NOT') is logic_mod.NOT
