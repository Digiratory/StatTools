from StatTools.generators.base_filter import FilteredArray
from StatTools.analysis.dfa import DFA

if __name__ == '__main__':
    threads = 12

    data = FilteredArray(h=0.8, length=1440).generate(n_vectors=1000, threads=threads)

    h_est = DFA(data, degree=2).parallel_2d(threads=threads, progress_bar=True)
