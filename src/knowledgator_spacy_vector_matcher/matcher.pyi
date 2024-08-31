from typing import (
    Any,
    Callable,
)

from spacy.vocab import Vocab
from spacy.matcher.matcher import Matcher
from numpy.typing import NDArray

class VectorMatcher(Matcher):
    def __init__(
        self,
        vocab: Vocab,
        validate: bool = ...,
        fuzzy_compare: Callable[[str, str, int], bool] = ...,
        similarity_compare: Callable[[NDArray[Any], NDArray[Any]], float] = ...,
        include_similarity_scores: bool=False,
    ) -> None: ...