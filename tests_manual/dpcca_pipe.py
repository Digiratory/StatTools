from StatTools.analysis.dpcca import dpcca
from StatTools.generators.base_filter import FilteredArray
from StatTools.generators.cholesky_transform import CorrelatedArray


if __name__ == '__main__':
    threads = 12

    # Эмпирические данные (траектории)
    data = FilteredArray(0.8, 1440, set_mean=10, set_std=3).generate(n_vectors=100, threads=threads)

    # Эмпирическое R
    s = [2 ** i for i in range(3, 10)]
    p, r, f, s = dpcca(data, pd=2, step=0.5, s=s, processes=threads)

    for s_i, s_val in enumerate(s):

        # FBM
        fbm = FilteredArray(h=0.8, length=1440).generate(n_vectors=100, threads=threads)

        # Холецкий с R
        chol_vectors = CorrelatedArray(data=fbm, threads=threads).create(corr_target=r[s_i]).to_numpy()

        # Еще раз DPCCA
        p_, r_, f_, s_ = dpcca(chol_vectors, pd=2, step=0.5, s=s, processes=threads)
