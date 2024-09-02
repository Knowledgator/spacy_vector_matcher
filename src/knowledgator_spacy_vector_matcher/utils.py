from typing import Any, List, Callable, Iterable, Union, Tuple

from spacy.tokens import Token, Span
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


def get_vector_from_token(t: Token) -> NDArray[Any]:
    return t._.embedding if t._.has("embedding") else t.vector


def get_vector_for_matching(
    nlp: Language, tokens: Iterable[Union[str, Tuple[str, int], Tuple[str, int, int], Token, Span]], resolver: Callable[[List[NDArray[Any]]], NDArray[Any]]=lambda x: np.mean(x, axis=0)
) -> NDArray[Any]:
    """Generate vector similar to provided tokens vectors using the resolver

    nlp (Language): The SpaCy pipeline
    tokens (Iterable[Union[str, Tuple[str, int], Token]]): Possible values:
        - (str): The text value of target token / span.
        - (str, int): The text where token can be found and its index.
        - (str, int, int): The text where span can be found and its start and end positions.
        - (Token): The SpaCy token.
        - (Span): The SpaCy span.
    resolver (Callable[[List[NDArray[Any]]], NDArray[Any]]): This function generates a vector that is similar to the provided token vectors.
    """
    vs = []
    for t in tokens:
        if isinstance(t, Token):
            rt = t
        elif isinstance(t, Span):
            vs.append(resolver([get_vector_from_token(rt) for rt in t]))
            continue
        elif isinstance(t, tuple):
            if len(t) not in [2, 3]:
                raise ValueError(f"Unexpected input: {t}")
            doc = nlp(t[0])
            if len(t) == 2:
                rt = doc[t[1]]
            elif len(t) == 3:
                vs.append(resolver([get_vector_from_token(rt) for rt in doc[t[1]:t[2]]]))
                continue
        else:
            doc = nlp(t)
            vs.append(resolver([get_vector_from_token(rt) for rt in doc]))
            continue
        vs.append(get_vector_from_token(rt))
    return resolver(vs)

    