# cython: binding=True, infer_types=True
# based on spacy
from typing import List

import warnings

import numpy as np
from cymem.cymem cimport Pool
from libc.stdint cimport int32_t
from murmurhash.mrmr cimport hash64
from spacy.attrs cimport ENT_IOB, NULL_ATTR
from spacy.tokens.token cimport Token
from spacy.attrs cimport ID
from spacy.matcher.matcher cimport Matcher, TokenPatternC, AttrValueC, IndexValueC, FINAL_ID, ONE
from spacy.typedefs cimport attr_t

from spacy.matcher.matcher import (
    _get_operators, _get_extensions, _predicate_cache_key, _get_extension_extra_predicates, 
    _RegexPredicate, _SetPredicate, _ComparisonPredicate, _FuzzyPredicate
)
from spacy.attrs import IDS
from spacy.errors import Errors, MatchPatternError, Warnings
from spacy.strings import get_string_id
from spacy.matcher.levenshtein import levenshtein_compare

from .schemas import validate_token_pattern
from .utils import cosine_similarity

cdef TokenPatternC* init_pattern(Pool mem, attr_t entity_id, object token_specs) except NULL: # from spaCy (here because of how cython works)
    pattern = <TokenPatternC*>mem.alloc(len(token_specs) + 1, sizeof(TokenPatternC))
    cdef int i, index
    for i, (quantifier, spec, extensions, predicates, token_idx) in enumerate(token_specs):
        pattern[i].quantifier = quantifier
        # Ensure attrs refers to a null pointer if nr_attr == 0
        if len(spec) > 0:
            pattern[i].attrs = <AttrValueC*>mem.alloc(len(spec), sizeof(AttrValueC))
        pattern[i].nr_attr = len(spec)
        for j, (attr, value) in enumerate(spec):
            pattern[i].attrs[j].attr = attr
            pattern[i].attrs[j].value = value
        if len(extensions) > 0:
            pattern[i].extra_attrs = <IndexValueC*>mem.alloc(len(extensions), sizeof(IndexValueC))
        for j, (index, value) in enumerate(extensions):
            pattern[i].extra_attrs[j].index = index
            pattern[i].extra_attrs[j].value = value
        pattern[i].nr_extra_attr = len(extensions)
        if len(predicates) > 0:
            pattern[i].py_predicates = <int32_t*>mem.alloc(len(predicates), sizeof(int32_t))
        for j, index in enumerate(predicates):
            pattern[i].py_predicates[j] = index
        pattern[i].nr_py = len(predicates)
        pattern[i].key = hash64(pattern[i].attrs, pattern[i].nr_attr * sizeof(AttrValueC), 0)
        pattern[i].token_idx = token_idx
    i = len(token_specs)
    # Use quantifier to identify final ID pattern node (rather than previous
    # uninitialized quantifier == 0/ZERO + nr_attr == 0 + non-zero-length attrs)
    pattern[i].quantifier = FINAL_ID
    pattern[i].attrs = <AttrValueC*>mem.alloc(1, sizeof(AttrValueC))
    pattern[i].attrs[0].attr = ID
    pattern[i].attrs[0].value = entity_id
    pattern[i].nr_attr = 1
    pattern[i].nr_extra_attr = 0
    pattern[i].nr_py = 0
    pattern[i].token_idx = -1
    return pattern

cdef class VectorMatcher(Matcher):
    """Match sequences of tokens, based on pattern rules.

    DOCS: https://spacy.io/api/matcher
    USAGE: https://spacy.io/usage/rule-based-matching
    """
    cdef public object _similarity

    def __init__(self, vocab, validate=True, *, fuzzy_compare=levenshtein_compare, similarity_compare=cosine_similarity): # TODO: as fuzzy compare add vector distance finder
        """Create the Matcher.

        vocab (Vocab): The vocabulary object, which must be shared with the
        validate (bool): Validate all patterns added to this matcher.
        fuzzy_compare (Callable[[str, str, int], bool]): The comparison method
            for the FUZZY operators.
        """
        super().__init__(vocab=vocab, validate=validate, fuzzy_compare=fuzzy_compare)
        self._similarity = similarity_compare


    def add(self, key, patterns, *, on_match=None, greedy: str = None): # from spaCy
        """Add a match-rule to the matcher. A match-rule consists of: an ID
        key, an on_match callback, and one or more patterns.

        If the key exists, the patterns are appended to the previous ones, and
        the previous on_match callback is replaced. The `on_match` callback
        will receive the arguments `(matcher, doc, i, matches)`. You can also
        set `on_match` to `None` to not perform any actions.

        A pattern consists of one or more `token_specs`, where a `token_spec`
        is a dictionary mapping attribute IDs to values, and optionally a
        quantifier operator under the key "op". The available quantifiers are:

        '!':      Negate the pattern, by requiring it to match exactly 0 times.
        '?':      Make the pattern optional, by allowing it to match 0 or 1 times.
        '+':      Require the pattern to match 1 or more times.
        '*':      Allow the pattern to zero or more times.
        '{n}':    Require the pattern to match exactly _n_ times.
        '{n,m}':  Require the pattern to match at least _n_ but not more than _m_ times.
        '{n,}':   Require the pattern to match at least _n_ times.
        '{,m}':   Require the pattern to match at most _m_ times.

        The + and * operators return all possible matches (not just the greedy
        ones). However, the "greedy" argument can filter the final matches
        by returning a non-overlapping set per key, either taking preference to
        the first greedy match ("FIRST"), or the longest ("LONGEST").

        Since spaCy v2.2.2, Matcher.add takes a list of patterns as the second
        argument, and the on_match callback is an optional keyword argument.

        key (Union[str, int]): The match ID.
        patterns (list): The patterns to add for the given key.
        on_match (callable): Optional callback executed on match.
        greedy (str): Optional filter: "FIRST" or "LONGEST".
        """
        errors = {}
        if on_match is not None and not hasattr(on_match, "__call__"):
            raise ValueError(Errors.E171.format(arg_type=type(on_match)))
        if patterns is None or not isinstance(patterns, List):  # old API
            raise ValueError(Errors.E948.format(arg_type=type(patterns)))
        if greedy is not None and greedy not in ["FIRST", "LONGEST"]:
            raise ValueError(Errors.E947.format(expected=["FIRST", "LONGEST"], arg=greedy))
        for i, pattern in enumerate(patterns):
            if len(pattern) == 0:
                raise ValueError(Errors.E012.format(key=key))
            if not isinstance(pattern, list):
                raise ValueError(Errors.E178.format(pat=pattern, key=key))
            if self.validate:
                errors[i] = validate_token_pattern(pattern)
        if any(err for err in errors.values()):
            raise MatchPatternError(key, errors)
        key = self._normalize_key(key)
        for pattern in patterns:
            try:
                specs = _preprocess_pattern(
                    pattern,
                    self.vocab,
                    self._extensions,
                    self._extra_predicates,
                    self._fuzzy_compare,
                    self._similarity # TODO: similarity resolver
                )
                self.patterns.push_back(init_pattern(self.mem, key, specs))
                for spec in specs:
                    for attr, _ in spec[1]:
                        self._seen_attrs.add(attr)
            except OverflowError, AttributeError:
                raise ValueError(Errors.E154.format()) from None
        self._patterns.setdefault(key, [])
        self._callbacks[key] = on_match
        self._filter[key] = greedy
        self._patterns[key].extend(patterns)


def _preprocess_pattern(token_specs, vocab, extensions_table, extra_predicates, fuzzy_compare, similarity):
    """This function interprets the pattern, converting the various bits of
    syntactic sugar before we compile it into a struct with init_pattern.

    We need to split the pattern up into four parts:
    * Normal attribute/value pairs, which are stored on either the token or lexeme,
        can be handled directly.
    * Extension attributes are handled specially, as we need to prefetch the
        values from Python for the doc before we begin matching.
    * Extra predicates also call Python functions, so we have to create the
        functions and store them. So we store these specially as well.
    * Extension attributes that have extra predicates are stored within the
        extra_predicates.
    * Token index that this pattern belongs to.
    """
    tokens = []
    string_store = vocab.strings
    for token_idx, spec in enumerate(token_specs):
        if not spec:
            # Signifier for 'any token'
            tokens.append((ONE, [(NULL_ATTR, 0)], [], [], token_idx))
            continue
        if not isinstance(spec, dict):
            raise ValueError(Errors.E154.format())
        ops = _get_operators(spec)
        attr_values = _get_attr_values(spec, string_store)
        extensions = _get_extensions(spec, string_store, extensions_table)
        predicates = _get_extra_predicates(spec, extra_predicates, vocab, fuzzy_compare, similarity) # TODO: similarity resolver
        for op in ops:
            tokens.append((op, list(attr_values), list(extensions), list(predicates), token_idx))
    return tokens


def _get_attr_values(spec, string_store): # can be removed
    attr_values = []
    for attr, value in spec.items():
        input_attr = attr
        if isinstance(attr, str):
            attr = attr.upper()
            if attr == '_':
                continue
            elif attr == "OP":
                continue
            if attr == "TEXT":
                attr = "ORTH"
            if attr == "IS_SENT_START":
                attr = "SENT_START"
            attr = IDS.get(attr) # TODO: can cause some troubles
        if isinstance(value, str):
            if attr == ENT_IOB and value in Token.iob_strings():
                value = Token.iob_strings().index(value)
            else:
                value = string_store.add(value)
        elif isinstance(value, bool):
            value = int(value)
        elif isinstance(value, int):
            pass
        elif isinstance(value, dict):
            continue
        else:
            raise ValueError(Errors.E153.format(vtype=type(value).__name__))
        if attr is not None:
            attr_values.append((attr, value))
        else:
            # should be caught in validation
            raise ValueError(Errors.E152.format(attr=input_attr))
    return attr_values


class _VectorPredicate: # custom predicate
    operators = ("EMBEDDING", "THRESHOLD")

    def __init__(
        self, i, attr, value, predicate=None, is_extension=False, vocab=None,  # TODO: remove predicate
        regex=False, fuzzy=None, fuzzy_compare=None, similarity=None
    ):
        self.i = i
        self.attr = attr
        self.predicate = predicate
        self.is_extension = is_extension
        self.key = _predicate_cache_key(self.attr, self.predicate, value)
        self.vocab = vocab
        self.similarity = similarity

        self.value = { # TODO: stupid! Maybe use pydantic
            "EMBEDDING": None,
            "THRESHOLD": None
        }

        # TODO: necessary only for I variant
        for k, v in value.items(): 
            k = k.upper()
            if k not in self.operators:
                raise ValueError(Errors.E126.format(good=self.operators, bad=k)) # TODO: change error formatting
            self.value[k] = v 
        
        # TODO: add value validation
        for k, v in self.value.items():
            if v is None:
                raise ValueError("Invalid value for: "+k+"; Value: "+str(v))
            if k == "EMBEDDING":
                self.value[k] = np.array(v)


    def __call__(self, Token token):
        if self.vocab.vectors.n_keys == 0:
            warnings.warn(Warnings.W007.format(obj="Token"))
        
        if token.vector_norm == 0:
            if not token.has_vector: # issue with token
                warnings.warn(Warnings.W008.format(obj="Token"))
            res = 0.
        else:
            res = self.similarity(token.vector, self.value["EMBEDDING"])
        
        return res >= self.value["THRESHOLD"]


def _get_extra_predicates(spec, extra_predicates, vocab, fuzzy_compare, similarity): # TODO: check for regex and other stuff 
                                                                                     # TODO: or wrapp and remove  
    predicate_types = {
        "REGEX": _RegexPredicate,
        "IN": _SetPredicate,
        "NOT_IN": _SetPredicate,
        "IS_SUBSET": _SetPredicate,
        "IS_SUPERSET": _SetPredicate,
        "INTERSECTS": _SetPredicate,
        "==": _ComparisonPredicate,
        "!=": _ComparisonPredicate,
        ">=": _ComparisonPredicate,
        "<=": _ComparisonPredicate,
        ">": _ComparisonPredicate,
        "<": _ComparisonPredicate,
        "FUZZY": _FuzzyPredicate,
        "FUZZY1": _FuzzyPredicate,
        "FUZZY2": _FuzzyPredicate,
        "FUZZY3": _FuzzyPredicate,
        "FUZZY4": _FuzzyPredicate,
        "FUZZY5": _FuzzyPredicate,
        "FUZZY6": _FuzzyPredicate,
        "FUZZY7": _FuzzyPredicate,
        "FUZZY8": _FuzzyPredicate,
        "FUZZY9": _FuzzyPredicate,
    }
    seen_predicates = {pred.key: pred.i for pred in extra_predicates}
    output = []
    for attr, value in spec.items():
        if isinstance(attr, str):
            if attr == "_":
                output.extend(
                    _get_extension_extra_predicates(
                        value, extra_predicates, predicate_types,
                        seen_predicates))
                continue
            elif attr.upper() == "OP":
                continue
            if attr.upper() == "TEXT":
                attr = "ORTH"
            attr = IDS.get(attr.upper()) # TODO: check for vector
        if isinstance(value, dict):
            output.extend(_get_extra_predicates_dict(attr, value, vocab, predicate_types,
                                                     extra_predicates, seen_predicates, fuzzy_compare=fuzzy_compare, similarity=similarity))
    return output


def _get_extra_predicates_dict(attr, value_dict, vocab, predicate_types,
                               extra_predicates, seen_predicates, regex=False, fuzzy=None, fuzzy_compare=None, similarity=None):
    output = []

    # TODO: custom logic for VECTOR / move out of here? 
    if attr == IDS.get("VECTOR"): # TODO: refactor
        cls = _VectorPredicate
        predicate = cls(len(extra_predicates), attr, value_dict, None, vocab=vocab,
                        regex=regex, fuzzy=fuzzy, fuzzy_compare=fuzzy_compare, similarity=similarity)
        if predicate.key in seen_predicates:
            output.append(seen_predicates[predicate.key])
        else:
            extra_predicates.append(predicate)
            output.append(predicate.i)
            seen_predicates[predicate.key] = predicate.i
        return output
    ##############################


    for type_, value in value_dict.items():
        type_ = type_.upper()
        cls = predicate_types.get(type_)
        if cls is None:
            warnings.warn(Warnings.W035.format(pattern=value_dict))
            # ignore unrecognized predicate type
            continue
        elif cls == _RegexPredicate:
            if isinstance(value, dict):
                # add predicates inside regex operator
                output.extend(_get_extra_predicates_dict(attr, value, vocab, predicate_types,
                                                         extra_predicates, seen_predicates,
                                                         regex=True))
                continue
        elif cls == _FuzzyPredicate:
            if isinstance(value, dict):
                # add predicates inside fuzzy operator
                fuzz = type_[len("FUZZY"):]  # number after prefix
                fuzzy_val = int(fuzz) if fuzz else -1
                output.extend(_get_extra_predicates_dict(attr, value, vocab, predicate_types,
                                                         extra_predicates, seen_predicates,
                                                         fuzzy=fuzzy_val, fuzzy_compare=fuzzy_compare))
                continue
        predicate = cls(len(extra_predicates), attr, value, type_, vocab=vocab,
                        regex=regex, fuzzy=fuzzy, fuzzy_compare=fuzzy_compare)
        # Don't create redundant predicates.
        # This helps with efficiency, as we're caching the results.
        if predicate.key in seen_predicates:
            output.append(seen_predicates[predicate.key])
        else:
            extra_predicates.append(predicate)
            output.append(predicate.i)
            seen_predicates[predicate.key] = predicate.i
    return output