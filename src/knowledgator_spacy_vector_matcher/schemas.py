from typing import (
    Any,
    List,
    Optional,
)

try:
    from pydantic.v1 import (
        BaseModel, Field
    )
except ImportError:
    from pydantic import (  # type: ignore
        BaseModel, Field
    )
from spacy.attrs import NAMES
from spacy.schemas import TokenPattern, validate

Embedding = List[Any] # TODO: make more type strict

class TokenPatternVector(BaseModel):
    embedding: Embedding
    threshold: float = 0

VectorValue = TokenPatternVector 

class VectorTokenPattern(TokenPattern):
    vector: Optional[VectorValue] = None

class VectorTokenPatternSchema(BaseModel):
    pattern: List[VectorTokenPattern] = Field(..., min_items=1)

    class Config:
        extra = "forbid"


def validate_token_pattern(obj: list) -> List[str]: # from spacy
    # Try to convert non-string keys (e.g. {ORTH: "foo"} -> {"ORTH": "foo"})
    get_key = lambda k: NAMES[k] if isinstance(k, int) and k < len(NAMES) else k
    if isinstance(obj, list):
        converted = []
        for pattern in obj:
            if isinstance(pattern, dict):
                pattern = {get_key(k): v for k, v in pattern.items()}
            converted.append(pattern)
        obj = converted
    return validate(VectorTokenPatternSchema, {"pattern": obj}) # change class for validation