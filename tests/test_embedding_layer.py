import logging

import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from knowledgator_spacy_vector_matcher import VectorMatcher, add_embeddings_layer, get_vector_for_matching

text = "Kyiv is a capital of Ukraine"

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()


def embedding_model(text: str):
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        embedding = model_output[0][:, 0][0]
    # normalize embedding
    return torch.nn.functional.normalize(embedding, p=2, dim=0).numpy()


def test_embedding_layer():
    nlp = spacy.load("en_core_web_sm")
    add_embeddings_layer(nlp, embedding_model)

    vector = get_vector_for_matching(nlp, ["Kyiv", "Madrid", "London"])
    pattern = [
        [{"VECTOR": {"embedding": vector, "threshold": 0.8}, "OP": "?"}],
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