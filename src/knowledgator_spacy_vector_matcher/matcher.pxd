from spacy.matcher.matcher cimport Matcher

cdef class VectorMatcher(Matcher):
    cdef public object _similarity
    cdef public object _embedding_model
    cdef public object _include_similarity_scores
