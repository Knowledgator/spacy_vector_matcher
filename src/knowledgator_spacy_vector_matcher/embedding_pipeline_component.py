from typing import Any, Callable

from numpy.typing import NDArray
from spacy.language import Language
from spacy.tokens import Token, Doc


def add_embeddings_layer(nlp: Language, embedding_model: Callable[[str], NDArray[Any]]): # TODO: refactor
    """Add embedding layer using provided model

    nlp (Language): The spaCy pipeline
    embedding_model (Callable[[str], NDArray[Any]]): The model that creates embedding. Embedding will be written to a Token extension ('embedding' key is used)
    """
    class StandaloneEmbedding:
        def __init__(self, nlp: Language, name: str):
            Token.set_extension("embedding", default=None)
            self.embedding_model = embedding_model


        def __call__(self, doc: Doc):
            for token in doc:
                token._.embedding = self.embedding_model(token.text)
            return doc
        
    @Language.factory("knowledgator_spacy_vector_matcher_embedding")
    def create_component(nlp, name):
        return StandaloneEmbedding(nlp, name)
    
    nlp.add_pipe("knowledgator_spacy_vector_matcher_embedding", last=True)