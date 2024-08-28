# based on spacy
from thinc.api import get_array_module

def vector_norm(vector): # TODO: add typing
    """The L2 norm of the token's vector representation.

    RETURNS (float): The L2 norm of the vector representation.
    """
    xp = get_array_module(vector)
    total = (vector ** 2).sum()
    return xp.sqrt(total) if total != 0. else 0.

def cosine_similarity(v1, v2): # TODO: add typing
    xp = get_array_module(v1)
    result = xp.dot(v1, v2) / (vector_norm(v1) * vector_norm(v2)) # TODO: find out vector norm
    # ensure we get a scalar back (numpy does this automatically but cupy doesn't)
    return result.item()