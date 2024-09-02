from typing import Any, List, Callable, Generator, Iterable, Union, Tuple
from enum import Enum

from spacy.tokens import Token, Span, Doc
from spacy.language import Language
import numpy as np
from numpy.typing import NDArray
from thinc.api import get_array_module

class ProcessingMode(Enum):
    CHAR = 0
    TOKEN = 1


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


def get_vectors_from_span(s: Union[Span, Doc]) -> List[NDArray[Any]]:
    return [get_vector_from_token(t) for t in s]


def get_vector_for_matching(
    nlp: Language, 
    tokens: Iterable[Union[
        str, 
        Tuple[str, int], 
        Tuple[str, int, int],
        Tuple[str, int, int, ProcessingMode], 
        Token, 
        Span
    ]], 
    resolver: Callable[[List[NDArray[Any]]], NDArray[Any]]=lambda x: np.mean(x, axis=0)
) -> NDArray[Any]:
    """Generate vector similar to provided tokens vectors using the resolver

    nlp (Language): The SpaCy pipeline
    tokens (Iterable[Union[str, Tuple[str, int], Token]]): Possible values:
        - (str): The text value of target token / span.
        - (str, int): The text where token can be found and its index.
        - (str, int, int): The text where span can be found and its chars start and end positions.
        - (str, int, int, ProcessingMode): The text where span can be found and its start and end positions:
            - chars positition with ProcessingMode.CHAR
            - tokens positition with ProcessingMode.TOKEN
        - (Token): The SpaCy token.
        - (Span): The SpaCy span.
    resolver (Callable[[List[NDArray[Any]]], NDArray[Any]]): This function generates a vector that is similar to the provided token vectors.
    """
    vs = []
    for t in tokens:
        if isinstance(t, Token):
            vs.append(get_vector_from_token(t))
        elif isinstance(t, Span):
            vs.append(resolver(get_vectors_from_span(t)))
        elif isinstance(t, tuple):
            TEXT, START, END, MODE = 0, 1, 2, 3

            doc = nlp(t[TEXT])
            if len(t) == 2:
                vs.append(get_vector_from_token(doc[t[START]]))
            elif len(t) == 3:
                vs.append(resolver(get_vectors_from_span(doc.char_span(t[START], t[END], alignment_mode="expand"))))
            elif len(t) == 4:
                if t[MODE] == ProcessingMode.TOKEN:
                    vs.append(resolver(get_vectors_from_span(doc[t[START]:t[END]])))
                elif t[MODE] == ProcessingMode.CHAR:
                    vs.append(resolver(get_vectors_from_span(doc.char_span(t[START], t[END], alignment_mode="expand"))))
                else:
                    raise ValueError("Invalid processing mode")
            else:
                raise ValueError(f"Unexpected input: {t}")
        else:
            doc = nlp(t)
            vs.append(resolver(get_vectors_from_span(doc)))
    return resolver(vs)

    