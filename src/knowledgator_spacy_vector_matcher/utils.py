from typing import Any, Iterable, Callable

from spacy.language import Language
import numpy as np
from numpy.typing import NDArray
from thinc.api import get_array_module


def vector_norm(vector: NDArray[Any]): # from spacy
    """The L2 norm of the token's vector representation.

    RETURNS (float): The L2 norm of the vector representation.
    """
    xp = get_array_module(vector)
    total = (vector ** 2).sum()
    return xp.sqrt(total) if total != 0. else 0.


def cosine_similarity(v1: NDArray[Any], v2: NDArray[Any]): # based on spacy
    xp = get_array_module(v1)
    result = xp.dot(v1, v2) / (vector_norm(v1) * vector_norm(v2))
    # ensure we get a scalar back (numpy does this automatically but cupy doesn't)
    return result.item()


def get_vector_for_matching(
    nlp: Language, tokens: Iterable[str], resolver: Callable[[Iterable[NDArray[Any]]], NDArray[Any]]=lambda x: np.add.reduce(x) / len(x)
) -> NDArray[Any]:
    """Generate vector similar to provided tokens vectors using the resolver

    nlp (Language): The SpaCy pipeline
    tokens (Iterable[str]): The text value of target tokens
    resolver (Callable[[Iterable[NDArray[Any]]], NDArray[Any]]): This function generates a vector that is similar to the provided token vectors.
    """
    vs = []
    for t in tokens:
        doc = nlp(t)
        assert len(doc) == 1, ValueError(f"Expected list of tokens! Get span: '{t}'. Tokens: {list(doc)}")
        vs.append(doc[0]._.embedding if doc[0]._.has("embedding") else doc[0].vector)
    return resolver(vs)

    