from typing import Any, List, Callable

import numpy as np
from numpy.typing import NDArray
from spacy.language import Language
from spacy.tokens import Token, Doc, Span
from flair.embeddings import TokenEmbeddings
from flair.data import Sentence


def add_embeddings_layer(
    nlp: Language, 
    embedding_model: TokenEmbeddings, 
    resolver: Callable[[List[NDArray[Any]]], NDArray[Any]]=lambda x: np.mean(x, axis=0)
): # TODO: refactor
    """Add embedding layer using provided model

    nlp (Language): The spaCy pipeline
    embedding_model (TokenEmbeddings): Initialized flair embeddings class. Embedding will be written to a Token extension ('embedding' key is used)
    resolver (Callable[[List[NDArray[Any]]], NDArray[Any]]): How to resolve differences in tokenization.
    """
    class StandaloneEmbedding:
        def __init__(self, nlp: Language, name: str):
            Token.set_extension("embedding", default=None)


        @classmethod
        def match_tokens(cls, span: Span, sentence: Sentence):
            i, j = 0, 0
            tokens = sentence.tokens
            while i < len(span) and j < len(tokens):
                ts = span[i]
                ts_start = ts.idx - span.start_char
                ts_end = ts_start + len(ts.text)

                tmp = []
                while j < len(tokens):
                    tf = tokens[j]
                    tf_start = tf.start_position
                    tf_end = tf.end_position
                    if tf_start >= ts_start and tf_end <= ts_end:
                        tmp.append(tf.embedding.cpu().numpy())
                    elif tf_start < ts_start:
                        pass
                    else:
                        break
                    j += 1
                # TODO: chek tmp value
                ts._.embedding = resolver(tmp)
                i += 1


        def __call__(self, doc: Doc):
            for span in doc.sents:
                s = Sentence(span.text)
                embedding_model.embed(s)
                self.match_tokens(span, s)
            return doc
        
    @Language.factory("knowledgator_spacy_vector_matcher_embedding")
    def create_component(nlp, name):
        return StandaloneEmbedding(nlp, name)
    
    nlp.add_pipe("knowledgator_spacy_vector_matcher_embedding", last=True)