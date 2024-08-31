# SpaCy extension for vector matching

## How to use

``` python
import spacy
from knowledgator_spacy_vector_matcher import VectorMatcher

nlp = spacy.load("en_core_web_sm")

text = "Kyiv is a capital of Ukraine"
vector = [-1.2164163, -0.90696347, 0.227983, 0.23504838, 0.8067142, -0.3233788, 0.5810419, 0.19609039, -1.0150186, -0.8773348, 1.7684244, -0.4627877, -0.6683071, 0.31202173, -1.6089913, -1.1123116, 0.7144386, 0.016228497, -1.134943, -0.013191134, -0.62124693, -0.39624658, -0.14702356, 0.78998005, -1.4407078, 0.104700685, 1.6932896, 0.77732193, -0.8665912, 0.22157007, 0.13423751, -0.89869094, 0.029444337, 0.1807715, -0.77379584, 0.79096514, -0.6899003, 0.3657312, 0.7586484, 3.339126, -1.0448824, 0.0007598698, 0.48853332, 1.8948479, -1.18613, 1.1770797, 0.109079815, 0.7690178, 0.99029005, -0.728811, -0.7904119, 1.2388561, -0.7435981, -1.2173172, -0.4164772, 0.48756135, 0.039435416, -0.05407724, 0.18628621, 0.14583746, 0.3073343, -0.5801205, -0.1814425, -0.6756009, 0.43325832, 0.48644775, -0.16097769, -0.013738453, 0.1800591, -1.252205, 0.46750036, -0.23839241, -0.28638372, -0.5620073, -0.36189649, -0.8900944, -0.1077009, -1.847414, 0.59084445, -0.063796505, -0.97242564, -0.52974236, -0.34687164, -0.3613115, 0.2679145, 1.728797, 0.011590779, 1.5414968, 0.60545504, 0.714426, -0.104911655, -0.08484535, 1.6626359, -1.003128, 0.14556935, 0.31219393]
pattern = [
    [{"VECTOR": {"embedding": vector, "threshold": 0.2}, "OP": "?"}],
]

matcher = VectorMatcher(nlp.vocab)
matcher.add("test", pattern)
doc = nlp(text)
matches = matcher(doc, as_spans=True)

for span in matches:
    logging.info(f'{span.text} -> {span.label_}')
```

## Generate vector

Create vector for matching pattern using predefined nlp

``` python
from knowledgator_spacy_vector_matcher import get_vector_for_matching

vector = get_vector_for_matching(nlp, ["Kyiv", "Madrid", "London"])
```

## Use embedding layer

Add extra custom layer for other embedding model 

``` python
import spacy
from knowledgator_spacy_vector_matcher import VectorMatcher, add_embeddings_layer

nlp = spacy.load("en_core_web_sm")
add_embeddings_layer(nlp, embedding_model)
```

or use trained models

## Get scores

Check similarity scores (vector pattern value, score)

``` python
matcher = VectorMatcher(nlp.vocab, include_similarity_scores=True) # include scores to token extensions
matcher.add("test", pattern)

doc = nlp(text)
matches = matcher(doc, as_spans=True)
for span in matches:
    logging.info(f'{span.text} -> {span.label_}')
    for token in span:
        logging.info(f'{token._.vector_match}') # list of tuples (vector pattern value, score)
```