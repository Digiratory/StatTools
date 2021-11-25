from StatTools.analysis.dpcca import dpcca, movmean
from StatTools.generators.base_filter import Filter
from matplotlib.pyplot import loglog, legend, show


if __name__ == '__main__':

    h = 1.5
    length = 2 ** 20
    s = [2 ** i for i in range(3, 20)]
    step = 0.5
    threads = 12

    x = Filter(h, length).generate()

    p1, r1, f1, s1 = dpcca(x, 2, step, s, processes=threads)
    loglog(s1, f1, label=f"Initial {len(x)}")

    for k in [2 ** i for i in [8, 10, 12]]:
        x2 = movmean(x, k)
        p2, r2, f2, s2 = dpcca(x2, 2, step, s, processes=threads)

        loglog(s2, f2, label=f"Movmean(k={k})")
    legend()
    show()