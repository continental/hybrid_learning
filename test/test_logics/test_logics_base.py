"""Tests for basic logical operations and parsing."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=not-callable
# pylint: disable=no-member
# pylint: disable=no-self-use
from collections import namedtuple
from typing import Dict, Any, Union, Callable, Set, List, Tuple, Sequence, Optional

import numpy as np
import pyparsing
import pytest
import torch

from hybrid_learning.fuzzy_logic import Merge, Logic, \
    lukasiewicz, boolean, goedel, product, MergeBuilder
from hybrid_learning.fuzzy_logic import quantifiers
from hybrid_learning.fuzzy_logic.predicates import arithmetic as ar
from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND as AND_, OR as OR_, NOT as NOT_
from hybrid_learning.fuzzy_logic.logic_base import FormulaParser


# Formula with new ID (identity) operator
class ID(Merge):
    """Dummy identity operation."""
    SYMB = "ID"
    ARITY = 1

    def operation(self, annotation_vals: Sequence) -> Any:
        """Return value of in_key unchanged."""
        return annotation_vals[0]

    def __repr__(self):
        return "ID()"


class TestLogicParsing:
    """Test the Boolean parser."""
    PARSER: FormulaParser = FormulaParser(boolean.BooleanLogic())

    @pytest.mark.parametrize('formula,parsed,parsed_str', [
        ("a&&b", boolean.AND('a', 'b'), None),
        ("a||b", boolean.OR('a', 'b'), None),
        ("~c", boolean.NOT('c'), None),
        ("a&&b||c", boolean.AND('a', boolean.OR('b', 'c')), None),
        ("a&&b||~c", boolean.AND('a', boolean.OR('b', boolean.NOT('c'))), None),
        ("&&||", None, None), ("~", None, None),
        ("a&&b||c", AND_("a", OR_("b", "c")), None),
        ("b||c&&a", AND_("a", OR_("b", "c")), None),
        # One may enclose expressions in brackets
        ("(a)&&(b||c)", AND_("a", OR_("b", "c")), "a&&b||c"),
        ("~a&&b||c", AND_(NOT_("a"), OR_("b", "c")), None),
        ("(~a)&&(b||c)", AND_(NOT_("a"), OR_("b", "c")), "~a&&b||c"),
        ("~(a&&b)||c", OR_(NOT_(AND_("a", "b")), "c"), None),
        ("(~a&&b)||c", OR_(AND_(NOT_("a"), "b"), "c"), None),
        # Diverse letters allowed
        ("a-b||c", OR_("a-b", "c"), None),
    ])
    def test_parse(self, formula: str, parsed: Merge, parsed_str: str):
        """Test main properties of parsing like idempotence"""
        parse = FormulaParser(boolean.Logic())
        # wrong formulas
        if parsed is None:
            with pytest.raises(pyparsing.ParseException):
                res = parse(formula)
                print(res)
            return
        parse_out = parse(formula)
        # parsing and usual calling interchangeable
        assert parsed == parse_out, "Error for {}".format(parse.__class__)
        # idempotence
        assert parse_out.to_infix_notation(sort_key=lambda _: 1, precedence=parse.logic) == \
               (parsed_str or formula), "Error for {}".format(parse.__class__)
        # out_key option
        assert parse(formula, out_key="c").out_key == "c", "Error for {}".format(parse.__class__)
        assert not parse(formula, overwrite=False).overwrite, "Error for {}".format(parse.__class__)

    @pytest.mark.parametrize('brackets,formula,parsed', [
        (('[', ']'), "[a&&b]||c", OR_(AND_('a', 'b'), 'c')),
        (('(', ']'), "(a&&b]||c", OR_(AND_('a', 'b'), 'c')),
        (('<<', '>>'), "<<a&&b>>||c", OR_(AND_('a', 'b'), 'c')),
    ])
    def test_brackets(self, brackets: Tuple[str, str], formula: str, parsed: Merge):
        parser = FormulaParser(boolean.Logic(), brackets=brackets)
        assert parser(formula) == parsed
        assert parser.to_str(formula, use_whitespace=False) == formula

    @pytest.mark.parametrize('should_succeed,allowed_chars,formula', [
        (False, 'a', 'a||b'),
        (False, pyparsing.alphanums, 'a-b || c'),
        (True, 'ab', 'a||b'),
        (True, 'a-b', 'a-b || b'),
        (True, 'a-b|', 'a-b || b'),
        (True, 'a-b|', 'a-b||b'),
    ])
    def test_allowed_chars(self, should_succeed: bool, allowed_chars: str, formula: str):
        parser = FormulaParser(logic=boolean.Logic(), allowed_chars=allowed_chars)
        if not should_succeed:
            with pytest.raises(pyparsing.ParseException):
                print(parser(formula))
        else:
            parser(formula)

    @pytest.mark.parametrize('is_normalized,op', [
        (True, boolean.AND(boolean.OR('a', 'b'), 'c')),
        (True, boolean.AND(boolean.OR('a', 'b'), 'c', boolean.NOT('d'))),
        (True, boolean.AND(boolean.OR(boolean.NOT('a'), 'b'), 'c')),
        (True, boolean.AND(boolean.OR('a', boolean.NOT('b')), 'c')),
        (False, boolean.OR(boolean.AND('a', 'b'), 'c')),  # AND inside of another operation
        (False, boolean.AND(boolean.AND('a', 'b'), 'c')),  # AND inside of AND
        (False, boolean.OR(boolean.OR('a', 'b'), 'c')),  # OR inside of OR
    ])
    def test_is_normal_form(self, op: Merge, is_normalized: bool):
        assert is_normalized == self.PARSER.is_normal_form(op)
        # empty logic raises
        with pytest.raises(ValueError):
            self.PARSER.functional_is_normal_form(op, [])

    def test_is_pure(self):
        """Test is_pure."""
        # Pure formula
        formula: Merge = boolean.AND(boolean.OR('a', 'b'), 'c', boolean.NOT('d'))
        assert self.PARSER.is_pure(formula)
        assert self.PARSER.logic.is_pure(formula)

        assert not self.PARSER.is_pure(boolean.AND(boolean.OR('a', 'b'), ID('c'), boolean.NOT('d')))

    def test_validate_logic(self):
        self.PARSER.functional_parse("a&&b||c", [boolean.AND, boolean.OR])
        with pytest.raises(ValueError):
            self.PARSER.functional_parse("a&&b||c", [boolean.AND, boolean.AND, boolean.OR])

    @pytest.mark.parametrize('inp,precedence,outp', [
        # Conjunctive normal form
        ("a&&b||c", [OR_, AND_], OR_(AND_("a", "b"), "c")),
        ("~a&&b||c", [AND_, OR_, NOT_], AND_(NOT_("a"), OR_("b", "c"))),
        # Disjunctive normal form
        ("a&&b||c", [AND_, OR_], AND_("a", OR_("b", "c"))),
        ("~a&&b||c", [OR_, AND_, NOT_], OR_(AND_(NOT_("a"), "b"), "c")),
        # NOT moved to middle
        ("~a&&b||c", [OR_, NOT_, AND_], OR_(NOT_(AND_("a", "b")), "c")),
        ("~a&&b||c", [AND_, NOT_, OR_], AND_(NOT_("a"), OR_("b", "c"))),
        ("~a||b&&c", [AND_, NOT_, OR_], AND_(NOT_(OR_("a", "b")), "c")),
    ])
    def test_operator_precedence(self, inp: str, precedence: List[Merge.__class__], outp: Merge):
        """Test whether the parsing respects operator precedence settings."""
        logic: Logic = boolean.Logic(precedence)
        assert self.PARSER.functional_parse(inp, precedence) == outp
        assert logic.parser()(inp) == outp
        assert FormulaParser(precedence)(inp) == outp
        assert FormulaParser(logic)(inp) == outp

        assert all(b in logic.operators for b in precedence)
        assert len(precedence) <= len(logic.operators)
        assert len(logic.operators) <= len(logic.DEFAULT_CONNECTIVES)

    @pytest.mark.parametrize('logic_cls,args,repr_out', [
        (lukasiewicz.Logic, dict(), "LukasiewiczLogic()"),
        (product.Logic, dict(), "ProductLogic()"),
        (goedel.Logic, dict(), "GoedelLogic()"),
        (boolean.Logic, dict(), "BooleanLogic()"),
        (product.Logic, dict(operators=[ID]), f"ProductLogic([{ID.__module__}.{ID.__name__}])"),
        (product.Logic, dict(AND=ID), f"ProductLogic(AND={ID.__module__}.{ID.__name__})"),
        (goedel.Logic, dict(AND=ID, OR=lukasiewicz.OR),
         f"GoedelLogic(AND={ID.__module__}.{ID.__name__}, "
         "OR=hybrid_learning.fuzzy_logic.tnorm_connectives.lukasiewicz.OR)"),
        (lukasiewicz.Logic, dict(operators=[MergeBuilder(ID, symb="op")], AND=ID, OR=lukasiewicz.OR),
         f"LukasiewiczLogic([MergeBuilder({ID.__module__}.{ID.__name__}, symb='op')], "
         f"AND={ID.__module__}.{ID.__name__})"),
    ])
    def test_repr(self, logic_cls, args: Dict[str, Any], repr_out: str):
        logic = logic_cls(**args)
        assert repr(logic) == repr_out


class TestMergeOperations:
    """Test the dictionary merge operations."""
    PARSER: FormulaParser = FormulaParser(boolean.Logic())

    @pytest.mark.parametrize('eq,first,second', [
        (True, boolean.AND('a', 'b'), boolean.AND('b', 'a')),
        (True, boolean.OR('a', 'b'), boolean.OR('b', 'a')),
        (True, boolean.AND(boolean.NOT('a'), 'b'), boolean.AND('b', boolean.NOT('a'))),
        (False, boolean.AND(boolean.NOT('a'), 'b'), boolean.AND('a', boolean.NOT('b'))),
    ])
    def test_equals(self, eq: bool, first: Merge, second: Merge):
        """Test __eq__"""
        assert eq == (first == second)

    @pytest.mark.parametrize('op', [
        boolean.AND('a', 'b'),
        boolean.OR('a', 'b'),
        boolean.AND(boolean.NOT('a'), 'b'),
        boolean.AND(boolean.NOT('a'), 'b'),
        boolean.NOT('a'),
        boolean.NOT(boolean.NOT('a')),
        boolean.AND.variadic_(), boolean.OR.variadic_(), boolean.NOT.variadic_(),
    ])
    def test_copy(self, op: Merge):
        copy = op.__copy__()
        assert copy is not op
        assert copy == op
        assert len(op.in_keys) == len(copy.in_keys)
        for copy_in_key, in_key in zip(copy.in_keys, op.in_keys):
            assert in_key == copy_in_key
            if isinstance(in_key, Merge):
                assert in_key is not copy_in_key

    @pytest.mark.parametrize('before,after,opts', [
        # commutative ops
        ("b&&a", "a&&b", {}), ("a&&b", "a&&b", {}),
        ("b||a", "a||b", {}), ("a||b", "a||b", {}),
        # non-commutative ops
        ("~a", "~a", {}),
        ("a>>b", "a>>b", {}), ("b>>a", "b>>a", {}),
        ("a<<b", "a<<b", {}), ("b<<a", "b<<a", {}),
        # variadic ops
        (boolean.AND.variadic_(), boolean.AND.SYMB, {}),
        (boolean.OR.variadic_(), boolean.OR.SYMB, {}),
        (boolean.NOT.variadic_(), boolean.NOT.SYMB, {}),
        # nested ops
        ("a&&b||c", "(b||c)&&a", {}),
        ("b&&d||~c||~e&&a", "((~c)||(~e)||d)&&a&&b", {}),
        (boolean.AND('c', boolean.OR('b', 'a')), "(a||b)&&c", {}),
        (boolean.AND(boolean.OR('a', 'b'), 'c'), "(a||b)&&c", {}),
        # with whitespace
        ("b&&d||~c||~e&&a", "((~c) || (~e) || d) && a && b", dict(use_whitespace=True)),
        (boolean.AND('c', boolean.OR('b', 'a')), "(a || b) && c", dict(use_whitespace=True)),
        (boolean.AND(boolean.OR('a', 'b'), 'c'), "(a || b) && c", dict(use_whitespace=True)),
        # with pretty symb
        ("b&&d||~c||~e&&a", "((NOTc)OR(NOTe)ORd)ANDaANDb", dict(use_pretty_op_symb=True)),
        (boolean.AND('c', boolean.OR('b', 'a')), "(aORb)ANDc", dict(use_pretty_op_symb=True)),
        (boolean.AND(boolean.OR('a', 'b'), 'c'), "(aORb)ANDc", dict(use_pretty_op_symb=True)),
        # different sorting
        ("Aa&&Aaaa||~Aaa||~Aaaaa&&A", "((~Aaaaa)||(~Aaa)||Aaaa)&&Aa&&A", dict(sort_key=lambda x: -len(x))),
        # different precedences
        (boolean.AND(boolean.OR('b', 'a'), boolean.NOT('c')), "(a||b)&&(~c)", {}),
        (boolean.AND(boolean.OR('b', 'a'), boolean.NOT('c')), "a||b&&~c",
         dict(precedence=[boolean.AND, boolean.OR, boolean.NOT])),
        (boolean.AND(boolean.OR('b', 'a'), boolean.NOT('c')), "(a||b)&&~c",
         dict(precedence=[boolean.OR, boolean.AND, boolean.NOT])),
        ("b&&d||~c||~e&&a", "a&&~e||b&&d||~c", dict(precedence=[boolean.OR, boolean.AND, boolean.NOT])),
        ("~b||a", "~a||b", dict(precedence=[boolean.NOT, boolean.OR, boolean.AND])),
    ])
    def test_infix_notation(self, before: Union[str, Merge], after: str, opts: Dict):
        """Test the to_infix_notation function"""
        if not isinstance(before, Merge):
            # Parse to merge operation:
            precedence = opts.get('precedence', self.PARSER.logic)
            before_op: Merge = FormulaParser.functional_parse(before, logic=precedence)
        else:
            before_op: Merge = before
        assert before_op.to_infix_notation(**opts) == after

    @pytest.mark.parametrize('op,str_out,pretty_str_out,repr_out', [
        (boolean.AND('a', 'b'), "a&&b", "aANDb", "AND('a', 'b')"),
        (boolean.AND('a', 'b', skip_none=True), "a&&b", "aANDb", "AND('a', 'b')"),
        (boolean.AND('a', 'b', skip_none=False), "a&&b", "aANDb", "AND('a', 'b', skip_none=False)"),
        (boolean.AND('a', 'b', overwrite=True), "a&&b", "aANDb", "AND('a', 'b')"),
        (boolean.AND('a', 'b', overwrite=False), "a&&b", "aANDb", "AND('a', 'b', overwrite=False)"),
        (boolean.AND('a', 'b', replace_none=None), "a&&b", "aANDb", "AND('a', 'b')"),
        (boolean.AND('a', 'b', replace_none=1), "a&&b", "aANDb", "AND('a', 'b', replace_none=1)"),
        (boolean.AND('a', 'b', out_key='blub'), "a&&b", "aANDb", "AND('a', 'b', out_key='blub')"),
        (boolean.AND('a', 'b', overwrite=False, skip_none=False, replace_none=1, out_key='blub'), "a&&b", "aANDb",
         "AND('a', 'b', out_key='blub', overwrite=False, replace_none=1, skip_none=False)"),
        (boolean.AND('b', 'a'), "a&&b", "aANDb", "AND('b', 'a')"),
        (boolean.OR('a', 'b'), "a||b", "aORb", "OR('a', 'b')"),
        (boolean.OR('b', 'a'), "a||b", "aORb", "OR('b', 'a')"),
        (boolean.NOT('a'), "~a", "NOTa", "NOT('a')"),
        (boolean.AND(boolean.NOT('a'), boolean.OR('b', boolean.NOT('c'))), "((~c)||b)&&(~a)", "NOTaANDNOTcORb",
         "AND(NOT('a'), OR('b', NOT('c')))"),
        (boolean.AND('a', 'b', out_key='c'), "a&&b", "aANDb", "AND('a', 'b', out_key='c')"),
        (boolean.AND.variadic_(), "&&", "AND", "AND(_variadic=True)"),
        (boolean.OR.variadic_(), "||", "OR", "OR(_variadic=True)"),
        (boolean.NOT.variadic_(), "~", "NOT", "NOT(_variadic=True)"),
        (boolean.AND.variadic_(out_key='c'), "&&", "AND", "AND(_variadic=True, out_key='c')"),
    ])
    def test_repr_and_str(self, op: Merge, str_out: str, pretty_str_out: str, repr_out: str):
        """Test the string and representation function."""
        assert str(op) == str_out
        assert op.to_pretty_str(precedence=self.PARSER.logic) == pretty_str_out
        if not op.is_variadic:
            assert self.PARSER.to_str(str_out, for_filenames=True) == pretty_str_out
            assert self.PARSER(str_out, **{k: v for k, v in op.settings.items() if
                                           k != 'in_keys' and k != 'symb'}) == op
        assert repr(op) == repr_out

    def test_init(self):
        """Test the different init arguments."""
        # more than one input value
        boolean.AND('a', 'b', 'c')
        boolean.OR('a', 'b', 'c')

        # non-conjunctive normal forms
        boolean.OR(boolean.AND('a', 'b'), 'c')
        boolean.AND(boolean.AND('a', 'b'), 'c')
        boolean.OR(boolean.OR('a', 'b'), 'c')
        boolean.NOT(boolean.OR(boolean.AND('a', 'b'), 'c'))

        # out_key, overwrite
        assert not boolean.AND('a', 'b', overwrite=False).overwrite
        assert not boolean.OR('a', 'b', overwrite=False).overwrite
        assert not boolean.NOT('a', overwrite=False).overwrite
        assert boolean.AND('a', 'b', out_key='c').out_key == 'c'
        assert boolean.OR('a', 'b', out_key='c').out_key == 'c'
        assert boolean.NOT('a', out_key='c').out_key == 'c'

        # NOT may not have more than one input
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            # pylint: disable=too-many-function-args
            boolean.NOT('a', 'b')
            # pylint: enable=too-many-function-args

        # Variadic init does not accept in_keys
        with pytest.raises(TypeError):
            boolean.AND.variadic_("a", "b")
        with pytest.raises(TypeError):
            boolean.OR.variadic_("a", "b")
        with pytest.raises(TypeError):
            boolean.NOT.variadic_("a")

    @pytest.mark.parametrize('merge_cls,inp,outp', [
        (boolean.AND, [1, 0., True], 0.),
        (boolean.AND, {"a": 1, "b": 0., "c": True}, 0.),
        (boolean.AND, [1, 1., True], 1.),
        (boolean.AND, {"a": 1, "b": 1., "c": True}, 1.),
        (boolean.OR, [1, 0., True], 1.),
        (boolean.OR, {"a": 1, "b": 0., "c": True}, 1.),
        (boolean.OR, [1, 1., True], 1.),
        (boolean.OR, {"a": 1, "b": 1., "c": True}, 1.),
        (boolean.NOT, [True], 0.),
        (boolean.NOT, {"a": 0.}, 1.),
    ])
    def test_variadic_(self, inp, outp, merge_cls):
        """Test the variadic_ initializer."""
        # Init
        m = merge_cls.variadic_()
        assert isinstance(m, merge_cls)
        assert m.is_variadic
        assert len(m.in_keys) == 0

        # Call
        m_outp = m(inp)
        assert np.allclose(m_outp, outp)

        # Wrong ARITY
        if m.ARITY == -1:
            with pytest.raises(IndexError):
                m([])
        else:
            too_many_args = {**inp, **{k: 1. for k in range(m.ARITY)}} if isinstance(inp, dict) \
                else (inp + [1.] * m.ARITY)
            with pytest.raises(IndexError):
                m(too_many_args)

    @pytest.mark.parametrize('op,additional_args', [
        (boolean.AND("a", "b"), dict(skip_none=False)),
        (boolean.AND("a", "b"), dict(replace_none=1)),
        (boolean.AND("a", "b"), dict(overwrite=False)),
        (boolean.AND("a", "b"), dict(out_key='blub')),
        (boolean.AND("a", "b"), dict(skip_none=False, replace_none=1, overwrite=False, out_key='blub')),
        (boolean.OR("a", "b", "c"), dict(skip_none=False, replace_none=1, overwrite=False, out_key='blub')),
        (boolean.NOT("a"), dict(skip_none=False, replace_none=1, overwrite=False, out_key='blub')),
    ])
    def test_with_(self, op: Merge, additional_args: Dict[str, Any]):
        builder = op.__class__.with_(**additional_args)
        new_op = builder(*op.in_keys)
        # Children should still be the same:
        assert new_op.to_str() == op.to_str()
        # Symbol should be the same:
        assert builder.SYMB == op.SYMB
        # But other settings should have changed:
        for key in additional_args:
            assert new_op.settings[key] == additional_args[key]

        assert repr(builder) == f"MergeBuilder({op.__class__.__module__}.{op.__class__.__name__}, " \
                                f"additional_args={repr(dict(sorted(additional_args.items())))})"

    @pytest.mark.parametrize('overwrite, formula_spec, inp, out_key, outp', [
        (True, "a&&b", {'a': 1, 'b': 0}, 'a', 0),
        (False, "a&&b", {'a': 1, 'b': 0}, 'a', None),
        ('noop', "a&&b", {'a': 1, 'b': 0}, 'a', 1),
        ('noop', "a&&b", {'a': 1, 'b': 0}, 'c', 0)
    ])
    def test_overwrite(self, overwrite, inp: Dict[str, Any], out_key: str, formula_spec: str, outp):
        # overwrite when specified
        formula: Merge = self.PARSER.parse(formula_spec, out_key=out_key, overwrite=overwrite)
        if outp is None:
            with pytest.raises(KeyError):
                formula(inp)
        else:
            out: Dict[str, Any] = formula(inp)
            assert out_key in out
            assert out[out_key] == outp

    def test_invalid_calls(self):
        apply: Callable = self.PARSER.apply
        # don't overwrite
        for spec in ("a&&b", "a||b", "~a"):
            with pytest.raises(KeyError):
                apply(spec, {'c': False}, out_key='c', overwrite=False)

        # masks are not broadcastable
        for spec in ("a&&b", "a||b"):
            with pytest.raises(ValueError):
                apply(spec, {'a': np.ones(3), 'b': np.zeros(2)})

    @pytest.mark.parametrize('spec,ann,out', [
        ("a&&b", {"a": np.ones([3, 3]), "b": np.zeros([3, 3])}, np.zeros([3, 3])),
        ("a||b", {"a": np.ones([3, 3]), "b": np.zeros([3, 3])}, np.ones([3, 3])),
        ("~a", {"a": np.ones([3, 3])}, np.zeros([3, 3])),
        # more sophisticated ones
        ("a&&b", {"a": np.array([0, 1, 0]), "b": np.array([1, 0, 1])}, np.zeros([3, 3])),
        ("a||b", {"a": np.array([0, 1, 0]), "b": np.array([1, 0, 1])}, np.ones([3, 3])),
        ("~a", {"a": np.array([0, 1, 0])}, np.array([1, 0, 1])),
        # mix of scalar and mask
        ("a&&b", {"a": np.array([0, 1, 0]), "b": 1}, np.array([0, 1, 0])),
        ("a||b", {"a": np.array([0, 1, 0]), "b": 1}, np.ones([3, 3])),
        ("~a", {"a": True}, False),
        ("a&&~b", {"a": np.array([0, 1, 0]), "b": 0}, np.array([0, 1, 0])),
        ("a||~b", {"a": np.array([0, 1, 0]), "b": 0}, np.ones([3, 3])),
        # more than 2 values
        ("a&&b&&c||b",
         {"a": np.array([0, 1, 1, 1]), "b": np.array([1, 0, 1, 1]),
          "c": np.array([1, 1, 0, 1])}, np.array([0, 0, 1, 1])),
    ])
    def test_call(self, spec: str, ann: Dict[str, Any], out: Dict[str, Any]):
        """Test the __call__ function on some samples"""
        orig_keys: Set[str] = set(ann.keys())
        operat: Merge = self.PARSER(spec, out_key='out')
        out_dict = operat(ann)
        # new keys added
        assert {*orig_keys, operat.out_key} == {*out_dict}, \
            "op: {}".format(repr(operat))
        # other values not changed
        for k in ann:
            assert out_dict[k] is ann[k]
        # correct out value
        assert np.allclose(out_dict['out'], out), "op: {}".format(repr(operat))

    @pytest.mark.parametrize('spec,ann,out_ann,keep_keys', [
        # No keep_keys
        ("a&&~b", {"a": 1, "b": 0}, {"a": 1, "b": 0, "(~b)&&a": 1}, None),
        ("a&&~b", {"a": 1, "b": 0}, {"a": 1, "b": 0, "(~b)&&a": 1}, []),
        ("~b&&(a||~b)", {"a": 1, "b": 0}, {"a": 1, "b": 0, "(~b)&&((~b)||a)": 1}, None),
        ("~b&&(a||~b)", {"a": 1, "b": 0}, {"a": 1, "b": 0, "(~b)&&((~b)||a)": 1}, []),
        # Some keep_keys
        ("~b&&(~a||~b)", {"a": 1, "b": 0}, {"a": 1, "b": 0, "~b": 1, "(~b)&&((~a)||(~b))": 1}, ['~b']),
        ("~b&&(~a||~b)", {"a": 1, "b": 0}, {"a": 1, "b": 0, "~a": 0, "(~b)&&((~a)||(~b))": 1}, ['~a']),
        ("~b&&(~a||~b)", {"a": 1, "b": 0}, {"a": 1, "b": 0, "~a": 0, "~b": 1, "(~b)&&((~a)||(~b))": 1}, ['~a', '~b']),
        # keep_keys not in self.all_out_keys
        ("a&&~b", {"a": 1, "b": 0}, {"a": 1, "b": 0, "(~b)&&a": 1}, ['c']),
    ])
    def test_keep_keys(self, spec: str, ann: Dict[str, Any], out_ann: Dict[str, Any], keep_keys: Sequence[str]):
        """Test different keep_keys settings."""
        orig_keys: Set[str] = set(ann.keys())
        operat: Merge = self.PARSER(spec) if isinstance(spec, str) else spec
        out_dict = operat(ann, keep_keys=keep_keys)
        # new keys added
        assert {*orig_keys, *set(keep_keys or []).intersection(operat.all_out_keys), operat.out_key} == {*out_dict}, \
            "op: {}\nkeep_keys: {}\nout: {}".format(repr(operat), keep_keys, out_dict)

    KEY_SPEC = namedtuple("KeySpec",
                          ['spec', 'children', 'consts', 'operation_keys',
                           'all_in_keys', 'all_out_keys'])

    @pytest.mark.parametrize('spec', [
        KEY_SPEC(spec="a&&~b", children=["~b"], consts={"a"},
                 operation_keys=['a', '~b'],
                 all_in_keys={'a', 'b'}, all_out_keys={'(~b)&&a', '~b'}),
        KEY_SPEC(spec="(a||b)&&(~c)&&d", children=["a||b", "~c"], consts={"d"},
                 operation_keys=['a||b', '~c', 'd'],
                 all_in_keys={'a', 'b', 'c', 'd'},
                 all_out_keys={'a||b', '~c', '(a||b)&&(~c)&&d'})
    ])
    def test_properties(self, spec: KEY_SPEC):
        """Test the properties around in_keys."""
        operat = self.PARSER(spec.spec)
        assert operat.children == [self.PARSER(c) for c in spec.children]
        assert operat.consts == spec.consts
        assert operat.operation_keys == spec.operation_keys
        assert operat.all_in_keys == spec.all_in_keys
        assert operat.all_out_keys == spec.all_out_keys


class TestArithmeticOperations:
    """Test custom merge operation ThreshedAt."""

    @pytest.mark.parametrize('formula,parsed', [
        ('a>b', ar.GreaterThan('a', 'b')),
        ('a||b > c', OR_('a', ar.GreaterThan('b', 'c'))),
        ('a||b>c', OR_('a', ar.GreaterThan('b', 'c'))),
        ('a<b', ar.LowerThan('a', 'b')),
        ('a||b < c', OR_('a', ar.LowerThan('b', 'c'))),
        ('a >= b', ar.GreaterEqualsThan('a', 'b')),
        ('a||b >= c', OR_('a', ar.GreaterEqualsThan('b', 'c'))),
        ('a <= b', ar.LowerEqualsThan('a', 'b')),
        ('a||b <= c', OR_('a', ar.LowerEqualsThan('b', 'c'))),
        ('a == b', ar.Equals('a', 'b')),
        ('a||b == c', OR_('a', ar.Equals('b', 'c'))),
    ])
    def test_parse(self, formula: str, parsed: Merge):
        logic = [*boolean.Logic(), *ar.ARITHMETIC_OP_PRECEDENCE]
        parser = FormulaParser(logic)
        assert parser.parse(formula) == parsed

    @pytest.mark.parametrize('formula,inp,thresh,expected', [
        # non-array inputs
        ('a>b', .1, .1, False), ('a>b', .5, 0, True), ('a>b', .5, 1, False),
        ('a>=b', .1, .1, True), ('a>=b', .5, 0, True), ('a>=b', .5, 1, False),
        ('a<b', .1, .1, False), ('a<b', .5, 0, False), ('a<b', .5, 1, True),
        ('a<=b', .1, .1, True), ('a<=b', .5, 0, False), ('a<=b', .5, 1, True),
        ('a==b', .5, .5, True), ('a==b', .3, .3, True), ('a==b', .5, 1, False), ('a==b', 1, .5, False),
        # standard cases
        ('a>b', np.array([1, 2, 3]), 3, np.array([False, False, False])),
        ('a>b', np.array([1, 2, 3]), 2, np.array([False, False, True])),
        ('a>b', np.array([1, 2, 3]), 1, np.array([False, True, True])),
        ('a>b', np.array([1, 2, 3]), .5, np.array([True, True, True])),
        ('a<b', np.array([1, 2, 3]), 3, np.array([True, True, False])),
        ('a<b', np.array([1, 2, 3]), 2, np.array([True, False, False])),
        ('a<b', np.array([1, 2, 3]), 1, np.array([False, False, False])),
        ('a<b', np.array([1, 2, 3]), .5, np.array([False, False, False])),
        ('a>=b', np.array([1, 2, 3]), 3, np.array([False, False, True])),
        ('a>=b', np.array([1, 2, 3]), 2, np.array([False, True, True])),
        ('a>=b', np.array([1, 2, 3]), 1, np.array([True, True, True])),
        ('a>=b', np.array([1, 2, 3]), .5, np.array([True, True, True])),
        ('a<=b', np.array([1, 2, 3]), 3, np.array([True, True, True])),
        ('a<=b', np.array([1, 2, 3]), 2, np.array([True, True, False])),
        ('a<=b', np.array([1, 2, 3]), 1, np.array([True, False, False])),
        ('a<=b', np.array([1, 2, 3]), .5, np.array([False, False, False])),
        ('a==b', np.array([1, 2, 3]), 3, np.array([False, False, True])),
        ('a==b', np.array([1, 2, 3]), 2, np.array([False, True, False])),
        ('a==b', np.array([1, 2, 3]), 1, np.array([True, False, False])),
        ('a==b', np.array([1, 2, 3]), .5, np.array([False, False, False])),
        # array threshold
        ('a>b', np.array([1, 2, 3]), np.array([.5, 2.5, 3]), np.array([True, False, False])),
        ('a>=b', np.array([1, 2, 3]), np.array([.5, 2.5, 3]), np.array([True, False, True])),
        ('a<b', np.array([1, 2, 3]), np.array([.5, 2.5, 3]), np.array([False, True, False])),
        ('a<=b', np.array([1, 2, 3]), np.array([.5, 2.5, 3]), np.array([False, True, True])),
        ('a==b', np.array([1, 2, 3]), np.array([.5, 2.5, 3]), np.array([False, False, True])),
        # broadcastable threshold
        ('a>b', np.array([[1, 2], [1, 2]]), np.array([.5, 2]), np.array([[True, False], [True, False]])),
        ('a>=b', np.array([[1, 2], [1, 2]]), np.array([.5, 2]), np.array([[True, True], [True, True]])),
        ('a<b', np.array([[.5, 2], [.5, 2]]), np.array([1, 2]), np.array([[True, False], [True, False]])),
        ('a<=b', np.array([[.5, 2], [.5, 2]]), np.array([1, 2]), np.array([[True, True], [True, True]])),
        ('a==b', np.array([[1, 2], [1, 2]]), np.array([.5, 2]), np.array([[False, True], [False, True]])),
    ])
    def test_gt(self, formula: str, inp: np.ndarray, thresh: Union[float, np.ndarray], expected: np.ndarray):
        logic = [*boolean.Logic(), *ar.ARITHMETIC_OP_PRECEDENCE]
        parser = FormulaParser(logic)
        op = parser.parse(formula)
        op_variadic = op.variadic_()
        self._execute_arithmetic_op(inp, thresh, expected, op, op_variadic)

    def _execute_arithmetic_op(self, inp_a: np.ndarray, inp_b: np.ndarray, expected: np.ndarray,
                               op: Merge, op_variadic: Merge):
        """Test arithmetic operation (variadic and normal) on given inputs and expected output."""
        err_msg = "Wrong output for operation {} and input {}:\nExpected: {}\nReceived: {}"
        # numpy
        op_in = {op.in_keys[0]: inp_a, op.in_keys[1]: inp_b}
        op_out = op(op_in)[op.out_key]
        assert np.array(op_out).dtype == np.bool
        assert np.allclose(op_out, expected), \
            err_msg.format(op, op_in, expected, op_out)
        assert np.allclose(op_variadic([inp_a, inp_b]), expected)
        # torch
        inp_t, expected_t = torch.tensor(inp_a), torch.tensor(expected)
        thresh_t = torch.tensor(inp_b) if isinstance(inp_b, np.ndarray) else inp_b
        op_in_t = {op.in_keys[0]: inp_t, op.in_keys[1]: thresh_t}
        op_out_t = op(op_in_t)[op.out_key]
        assert op_out_t.dtype == torch.bool
        assert (op_out_t == expected_t).all()
        assert (op_variadic([inp_t, thresh_t]) == expected_t).all()


class TestQuantifier:
    """Test the quantifiers."""

    @pytest.mark.parametrize('formula_spec,inp,expected,dim,reduction', [
        # no dim given
        ('All inp', [[1, 2], [3, 4]], 1, None, 'min'),
        ('All inp', [[1, 2], [3, 4]], 4, None, 'max'),
        ('All inp', [[1, 2], [3, 4]], 2.5, None, 'mean'),
        ('Any inp', [[1, 2], [3, 4]], 1, None, 'min'),
        ('Any inp', [[1, 2], [3, 4]], 4, None, 'max'),
        ('Any inp', [[1, 2], [3, 4]], 2.5, None, 'mean'),
        # one dim
        ('All inp', [[1, 2], [3, 4]], [1, 3], 1, 'min'),
        ('All inp', [[1, 2], [3, 4]], [1, 2], 0, 'min'),
        ('All inp', [[1, 2], [3, 4]], [2, 4], 1, 'max'),
        ('All inp', [[1, 2], [3, 4]], [3, 4], 0, 'max'),
        ('All inp', [[1, 2], [3, 4]], [1.5, 3.5], 1, 'mean'),
        ('All inp', [[1, 2], [3, 4]], [2, 3], 0, 'mean'),
        ('Any inp', [[1, 2], [3, 4]], [1, 3], 1, 'min'),
        ('Any inp', [[1, 2], [3, 4]], [1, 2], 0, 'min'),
        ('Any inp', [[1, 2], [3, 4]], [2, 4], 1, 'max'),
        ('Any inp', [[1, 2], [3, 4]], [3, 4], 0, 'max'),
        ('Any inp', [[1, 2], [3, 4]], [1.5, 3.5], 1, 'mean'),
        ('Any inp', [[1, 2], [3, 4]], [2, 3], 0, 'mean'),
        # two dims
        ('All inp', [[[1], [2]], [[3], [4]]], [1, 3], [1, 2], 'min'),
        ('All inp', [[[1], [2]], [[3], [4]]], [2, 4], (1, 2), 'max'),
        ('All inp', [[[1], [2]], [[3], [4]]], [1.5, 3.5], [1, 2], 'mean'),
        ('Any inp', [[[1], [2]], [[3], [4]]], [1, 3], [1, 2], 'min'),
        ('Any inp', [[[1], [2]], [[3], [4]]], [2, 4], (1, 2), 'max'),
        ('Any inp', [[[1], [2]], [[3], [4]]], [1.5, 3.5], [1, 2], 'mean'),
        # custom reduction, no dim
        ('All inp', [[1, .5, .5], [1, .5, 1]], .125, None, product.AND.variadic_()),
        ('All inp', [[1, .5, .5], [1, .5, 0]], 1, None, lukasiewicz.OR.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], 0, None, product.AND.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], 1, None, lukasiewicz.OR.variadic_()),
        # custom reduction, one dim
        ('All inp', [[1, 1, 1], [1, 0, 0]], [1, 0, 0], 0, boolean.AND.variadic_()),
        ('All inp', [[1, 1, 1], [1, 0, 0]], [1, 0], 1, boolean.AND.variadic_()),
        ('All inp', [[1, .5, .5], [1, .5, 0]], [1, .25, 0], 0, product.AND.variadic_()),
        ('All inp', [[1, .5, .5], [1, .5, 0]], [.25, 0], 1, product.AND.variadic_()),
        ('All inp', [[1, .5, .5], [1, .5, 0]], [1, 1, .5], 0, lukasiewicz.OR.variadic_()),
        ('All inp', [[1, .5, .5], [1, .5, 0]], [1, 1], 1, lukasiewicz.OR.variadic_()),
        ('Any inp', [[1, 1, 1], [1, 0, 0]], [1, 0, 0], 0, boolean.AND.variadic_()),
        ('Any inp', [[1, 1, 1], [1, 0, 0]], [1, 0], 1, boolean.AND.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], [1, .25, 0], 0, product.AND.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], [.25, 0], 1, product.AND.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], [1, 1, .5], 0, lukasiewicz.OR.variadic_()),
        ('Any inp', [[1, .5, .5], [1, .5, 0]], [1, 1], 1, lukasiewicz.OR.variadic_()),
        # custom reduction, several dims
        ('All inp', [[[1], [.5], [.5]], [[1], [.5], [1]]], [.125], [0, 1], product.AND.variadic_()),
        ('All inp', [[[1], [.5], [.5]], [[1], [.5], [0]]], [1, .25, 0], [0, 2], product.AND.variadic_()),
        ('All inp', [[[1], [.5], [.5]], [[1], [.5], [0]]], [.25, 0], [1, 2], product.AND.variadic_()),
        ('Any inp', [[[1], [.5], [.5]], [[1], [.5], [1]]], [.125], [0, 1], product.AND.variadic_()),
        ('Any inp', [[[1], [.5], [.5]], [[1], [.5], [0]]], [1, .25, 0], [0, 2], product.AND.variadic_()),
        ('Any inp', [[[1], [.5], [.5]], [[1], [.5], [0]]], [.25, 0], [1, 2], product.AND.variadic_()),
        # empty tensor
        ('Any inp', [], [], -4, 'max'),
        ('All inp', [], [], -4, 'min'),
        ('Any inp', [], [], -4, product.AND.variadic_()),
        ('All inp', [], [], -4, lukasiewicz.OR.variadic_()),
    ])
    def test_quantifier(self, formula_spec: str, inp: Sequence, expected: Sequence,
                        dim: Union[int, Sequence[int]], reduction: Union[str, Merge]):
        logic = boolean.Logic([quantifiers.ALL.with_(dim=dim, reduction=reduction),
                               quantifiers.ANY.with_(dim=dim, reduction=reduction)])
        formula: Merge = logic.parser()(formula_spec)
        assert sorted(formula.in_keys) == sorted(['inp'])

        # numpy
        out = formula(dict(inp=np.array(inp)))[formula.out_key]
        assert isinstance(out, (np.ndarray, np.number, float, int, bool)), "Out is of type {}".format(type(out))
        assert np.allclose(out, expected)

        # torch
        out = formula(dict(inp=torch.as_tensor(inp)))[formula.out_key]
        assert isinstance(out, torch.Tensor), "Out is of type {}".format(type(out))
        assert np.allclose(out, expected)

    @pytest.mark.parametrize('formula_spec,inp,cond,expected,dim,raises', [
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True]], [[2], [4]], None, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[True, False]], [[1], [3]], None, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[True], [False]], [[1, 2]], None, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], None, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], None, False),
        # non-boolean (but binary) cond
        ('inp Where cond', [[1, 2], [3, 4]], [[0], [1]], [[3, 4]], None, False),
        # with a bit of other logics
        ('inp Where (~cond)', [[1, 2], [3, 4]], [[True], [False]], [[3, 4]], None, False),
        ('inp Where (~cond)', [[1, 2], [3, 4]], [[False], [True]], [[1, 2]], None, False),
        # wrong cond format
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True], [False, True]], [[3, 4]], None, ValueError),
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True], [False, True]], [[3, 4]], None, ValueError),
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True], [False, True]], [[3, 4]], 0, ValueError),
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True], [False, True]], [[3, 4]], 1, ValueError),
        # wrong dim format
        ('inp Where cond', [[1, 2], [3, 4]], [[False, True]], [[3, 4]], "blub", ValueError),
        # wrong inp format
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], 2, ValueError),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], 3, ValueError),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], -3, ValueError),
        # fixed dim
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], 0, False),
        ('inp Where cond', [[1, 2], [3, 4]], [False, True], [[3, 4]], 0, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[2], [4]], 1, False),
        ('inp Where cond', [[1, 2], [3, 4]], [False, True], [[2], [4]], 1, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[3, 4]], -2, False),
        ('inp Where cond', [[1, 2], [3, 4]], [False, True], [[3, 4]], -2, False),
        ('inp Where cond', [[1, 2], [3, 4]], [[False], [True]], [[2], [4]], -1, False),
        ('inp Where cond', [[1, 2], [3, 4]], [False, True], [[2], [4]], -1, False),
    ])
    def test_where(self, formula_spec: str, inp: Sequence, cond: Sequence, expected: Sequence,
                   raises, dim: Optional[int]):
        logic = boolean.Logic([quantifiers.WHERE.with_(dim=dim)])
        if raises:
            with pytest.raises(raises):
                formula: Merge = logic.parser()(formula_spec)
                formula(dict(inp=np.array(inp), cond=cond))
        else:
            # torch
            formula: Merge = logic.parser()(formula_spec)
            assert sorted(formula.all_in_keys) == sorted(['inp', 'cond'])
            out = formula(dict(inp=torch.as_tensor(inp), cond=torch.as_tensor(cond)))[formula.out_key]
            assert isinstance(out, torch.Tensor), "Out is of type {}".format(type(out))
            assert np.allclose(out, expected), "Wrong output for dim={} and\ncond size {}\ntens size {}" \
                .format(dim, np.array(cond).shape, np.array(inp).shape)
