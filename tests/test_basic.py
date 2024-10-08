import logging

import numpy as np
import spacy
from knowledgator_spacy_vector_matcher import VectorMatcher, ProcessingMode, get_vector_for_matching



def test_match():
    text = "Kyiv is a capital of Ukraine"
    vector = np.array([-1.2164163, -0.90696347, 0.227983, 0.23504838, 0.8067142, -0.3233788, 0.5810419, 0.19609039, -1.0150186, -0.8773348, 1.7684244, -0.4627877, -0.6683071, 0.31202173, -1.6089913, -1.1123116, 0.7144386, 0.016228497, -1.134943, -0.013191134, -0.62124693, -0.39624658, -0.14702356, 0.78998005, -1.4407078, 0.104700685, 1.6932896, 0.77732193, -0.8665912, 0.22157007, 0.13423751, -0.89869094, 0.029444337, 0.1807715, -0.77379584, 0.79096514, -0.6899003, 0.3657312, 0.7586484, 3.339126, -1.0448824, 0.0007598698, 0.48853332, 1.8948479, -1.18613, 1.1770797, 0.109079815, 0.7690178, 0.99029005, -0.728811, -0.7904119, 1.2388561, -0.7435981, -1.2173172, -0.4164772, 0.48756135, 0.039435416, -0.05407724, 0.18628621, 0.14583746, 0.3073343, -0.5801205, -0.1814425, -0.6756009, 0.43325832, 0.48644775, -0.16097769, -0.013738453, 0.1800591, -1.252205, 0.46750036, -0.23839241, -0.28638372, -0.5620073, -0.36189649, -0.8900944, -0.1077009, -1.847414, 0.59084445, -0.063796505, -0.97242564, -0.52974236, -0.34687164, -0.3613115, 0.2679145, 1.728797, 0.011590779, 1.5414968, 0.60545504, 0.714426, -0.104911655, -0.08484535, 1.6626359, -1.003128, 0.14556935, 0.31219393])
    pattern = [
        [{"VECTOR": {"embedding": vector, "threshold": 0.2}, "OP": "?"}],
    ]


    nlp = spacy.load("en_core_web_sm")
    matcher = VectorMatcher(nlp.vocab, include_similarity_scores=True)
    matcher.add("test", pattern)

    doc = nlp(text)
    matches = matcher(doc, as_spans=True)
    # assert len(matches) == 1
    for span in matches:
        logging.info(f'{span.text} -> {span.label_}')
        for token in span:
            logging.info(f'{token._.vector_match}')


def test_vector_generation_with_context_dependence():
    nlp = spacy.load("en_core_web_sm")

    text_sample = "Kyiv is a capital of Ukraine"
    vector = get_vector_for_matching(
        nlp,
        [
            "Kyiv",
            nlp(text_sample)[0],
            ("I love my native city Kyiv, wich is one of the most beautiful cities in Europe", 5)
        ]
    )
    pattern = [
        [{"VECTOR": {"embedding": vector, "threshold": 0.65}, "OP": "?"}],
    ]

    matcher = VectorMatcher(nlp.vocab, include_similarity_scores=True)
    matcher.add("test", pattern)

    text = "I traveled to Ukraine and visited Kyiv, which is a capital of this country."
    doc = nlp(text)
    matches = matcher(doc, as_spans=True)
    assert len(matches) == 1
    for span in matches:
        logging.info(f'{span.text} -> {span.label_}')
        for token in span:
            logging.info(f'{token._.vector_match}')


def test_span_processing():
    nlp = spacy.load("en_core_web_sm")

    matcher_span = VectorMatcher(nlp.vocab, include_similarity_scores=True)
    matcher_span.add("span", [
        [{"OP": "{1}"}, {"LEMMA": "be"}, {"OP": "?"}, {"TEXT": "capital"}]
    ])

    
    text_sample1 = "Warshaw is a capital of Poland."
    doc1 = nlp(text_sample1)
    to_process = [
        (text_sample1, start, end, ProcessingMode.TOKEN) for _, start, end in matcher_span(doc1)
    ]
    
    text_sample2 = "London is a capital of England."
    doc2 = nlp(text_sample2)
    to_process.extend((text_sample2, s.start_char, s.end_char) for s in matcher_span(doc2, as_spans=True))

    text_sample3 = "Washington is a capital of the United States."
    doc3 = nlp(text_sample3)
    to_process.extend(matcher_span(doc3, as_spans=True))

    text_sample4 = "Paris is a capital of France."
    doc4 = nlp(text_sample4)
    to_process.extend((text_sample4, s.start_char, s.end_char, ProcessingMode.CHAR) for s in matcher_span(doc4, as_spans=True))


    vector = get_vector_for_matching(
        nlp,
        to_process
    )
    pattern = [
        [{"VECTOR": {"embedding": vector, "threshold": 0.25}, "OP": "?"}],
    ]
    
    matcher_res = VectorMatcher(nlp.vocab, include_similarity_scores=True)
    matcher_res.add("test", pattern)

    text = "Kyiv is a capital of Ukraine"
    doc = nlp(text)
    matches = matcher_res(doc, as_spans=True)

    assert " ".join((span.text for span in matches)) == "Kyiv is a capital"