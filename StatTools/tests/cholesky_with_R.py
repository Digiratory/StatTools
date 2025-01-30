from StatTools.analysis.dpcca import dpcca
from StatTools.generators.base_filter import FilteredArray
from StatTools.generators.cholesky_transform import CorrelatedArray
from StatTools.auxiliary import PearsonParallel


if __name__ == '__main__':

    threads = 4

    empirical_data = FilteredArray(1.5, 2**10).generate(n_vectors=2**7, threads=threads)        # траектории
    P, R, F, S = dpcca(empirical_data, 2, 0.5, [2**i for i in range(3, 10)], processes=threads)

    for i, s in enumerate(S):
        correlated_vectors = CorrelatedArray(data=empirical_data, threads=threads).create(corr_target=R[i])

        corr = PearsonParallel(correlated_vectors).create_matrix(threads=threads)
