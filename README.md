This library allows to create and process long-term dependent datasets.

Installation:

    python setup.py install

Basis usage:

    1. To create a simple dataset with given Hurst parameter:

        from StatTools.filters import FilteredArray

        h = 0.8                 # choose Hurst parameter
        total_vectors = 1000    # total number of vectors in output
        vectors_length = 1440   # each vector's length 
        t = 8                   # threads in use during computation

        correlated_vectors = Filter(h, vectors_length).generate(n_vectors=total_vectors,
                                                                threads=t, progress_bar=True)

Current stable version: 1.1.1