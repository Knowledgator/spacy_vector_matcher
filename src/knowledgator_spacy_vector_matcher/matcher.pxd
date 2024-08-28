from spacy.matcher.matcher cimport Matcher

cdef class VectorMatcher(Matcher):
    cdef public object _similarity
