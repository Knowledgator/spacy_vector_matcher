from typing import (
    List,
    Optional,
)

try:
    from pydantic.v1 import (
        BaseModel, Field, validator
    )
except ImportError:
    from pydantic import (  # type: ignore
        BaseModel, Field, validator
    )
from spacy.attrs import NAMES
from spacy.schemas import TokenPattern, validate
import numpy as np

Embedding = np.ndarray

class Vector(BaseModel):
    embedding: Embedding
    threshold: float = 0

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        alias_generator = lambda value: value.upper()
        arbitrary_types_allowed = True


    @validator("*", pre=True, allow_reuse=True)
    def raise_for_none(cls, v):
        if v is None:
            raise ValueError("None / null is not allowed")
        return v

VectorValue = Vector 

class VectorTokenPattern(TokenPattern):
    vector: Optional[VectorValue] = None


class VectorTokenPatternSchema(BaseModel):
    pattern: List[VectorTokenPattern] = Field(..., min_items=1)


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