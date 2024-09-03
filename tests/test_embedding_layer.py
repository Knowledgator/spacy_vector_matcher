import logging

import spacy
from flair.embeddings import TransformerWordEmbeddings
from knowledgator_spacy_vector_matcher import VectorMatcher, add_embeddings_layer, get_vector_for_matching

text = "Kyiv is a capital of Ukraine"


def test_embedding_layer():
    nlp = spacy.load("en_core_web_sm")
    add_embeddings_layer(nlp, TransformerWordEmbeddings(model="BAAI/bge-large-zh-v1.5"))

    vector = get_vector_for_matching(nlp, ["Kyiv", "Madrid", "London"])
    pattern = [
        [{"VECTOR": {"embedding": vector, "threshold": 0.6}, "OP": "?"}],
    ]

    matcher = VectorMatcher(nlp.vocab, include_similarity_scores=True)
    matcher.add("test", pattern)

    doc = nlp(text)
    matches = matcher(doc, as_spans=True)
    assert len(matches) == 1
    for span in matches:
        logging.info(f'{span.text} -> {span.label_}')
        for token in span:
            logging.info(f'{token._.vector_match}')