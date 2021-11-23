import numpy
from tqdm import tqdm
from multiprocessing import Pool, Value, Array, Lock
from ctypes import c_double
from contextlib import closing
from threading import Thread
from StatTools.analysis.dfa import bar_manager
from functools import partial


class GammaReplacement():

    def __init__(self, dataset, set_mean=20, set_std=6):

        if set_mean < 0:
            error_str = "\tGammaReplacement Error: Mean value is not supposed to be negative!"
            raise NameError(error_str)

        if set_std ==0 or set_mean == 0:
            error_str = "\tGammaReplacement Error: Zero mean or std!"
            raise NameError(error_str)

        if not isinstance(dataset, type(numpy.array([]))):
            self.dataset = numpy.array(dataset)

        else:
            self.dataset = dataset

        self.set_std = set_std
        self.set_mean = set_mean

    @staticmethod
    def rank_replacement_with_gamma(initial_vector):
        sorted_x1 = numpy.sort(initial_vector)

        var_normal = numpy.var(initial_vector, ddof=1)
        mean_normal = numpy.mean(initial_vector)

        theta = var_normal / mean_normal
        k = mean_normal / (var_normal / mean_normal)
        validation_vector = numpy.random.mtrand.gamma(k, theta, len(initial_vector))

        sorted_x2 = numpy.sort(validation_vector)

        # @njit(cache=True)
        def rank_replacement_core(x1, sorted_x1, sorted_x2):
            for i in range(len(x1)):
                get_index_in_sorted_x1 = numpy.where(sorted_x1 == x1[i])[0][0]
                x1[i] = sorted_x2[get_index_in_sorted_x1]
            return x1

        x1 = rank_replacement_core(initial_vector, sorted_x1, sorted_x2)

        return x1

    def perform(self, threads=1, progress_bar=False):

        if threads == 1:

            if self.dataset.size > pow(10, 6):
                print("\tGamaReplacement Warning: Given the size of input dataset it'd be faster to use more threads . . .")

            if progress_bar:
                bar = tqdm(desc="GammaReplacment", total=len(self.dataset), leave=False, position=0)

            one_dim = False

            if self.dataset.ndim == 1:

                self.dataset = [self.dataset]
                one_dim = True

            for v, vector in enumerate(self.dataset):
                self.dataset[v] = self.dataset[v] * (self.set_std / numpy.std(self.dataset[v], ddof=1))
                self.dataset[v] = self.dataset[v] + (-numpy.mean(self.dataset[v]) + self.set_mean)
                self.dataset[v] = self.rank_replacement_with_gamma(self.dataset[v])
                self.dataset[v] = self.dataset[v] * (self.set_std / numpy.std(self.dataset[v], ddof=1))
                self.dataset[v] = self.dataset[v] + (-numpy.mean(self.dataset[v]) + self.set_mean)
                a = numpy.std(self.dataset[v], ddof=1)
                if progress_bar:
                    bar.update(1)

            if one_dim:
                return self.dataset[0]

            return self.dataset

        else:

            vectors_indices = numpy.linspace(0, len(self.dataset) - 1, len(self.dataset), dtype=int)
            vectors_by_threads = numpy.array_split(vectors_indices, threads)

            shared_initial_dataset = Array(c_double, self.dataset.size)
            vectors_num = len(self.dataset)
            vectors_length = len(self.dataset[0])

            numpy.copyto(numpy.frombuffer(shared_initial_dataset.get_obj()).reshape((vectors_num, vectors_length)),
                         self.dataset)

            bar_value = Value('i', 0)
            stop_bit = Value('i', 0)
            lock = Lock()

            if progress_bar:
                bar = Thread(target=bar_manager, args=('GammaReplacement', vectors_num, bar_value, lock, 'total', stop_bit))
                bar.start()

            with closing(Pool(processes=threads, initializer=self.global_initializer, initargs=(shared_initial_dataset,
                                                                                                bar_value, lock))) as pool:
                pool.map(partial(self.processing_vectors, set_mean=self.set_mean, set_std=self.set_std, vectors_num=vectors_num,
                                 vectors_length=vectors_length, function=self.rank_replacement_with_gamma), vectors_by_threads)

            stop_bit.value += 1

            output_array = numpy.frombuffer(shared_initial_dataset.get_obj()).reshape((vectors_num, vectors_length))

            return output_array

    @staticmethod
    def global_initializer(shared_init_data, b_val, b_lk):
        global SHARED_VECTORS
        global BAR_VALUE
        global BAR_LOCK
        SHARED_VECTORS = shared_init_data
        BAR_VALUE = b_val
        BAR_LOCK = b_lk


    @staticmethod
    def processing_vectors(indices, set_mean, set_std, vectors_num, vectors_length, function):

        def get_pointer():
            return numpy.frombuffer(SHARED_VECTORS.get_obj(), dtype=c_double).reshape((vectors_num, vectors_length))

        for v in indices:
            vector = get_pointer()[v]
            vector = vector * (set_std / numpy.std(vector, ddof=1))
            vector = vector + numpy.abs(numpy.mean(vector) - set_mean)
            vector = function(vector)
            vector = vector * (set_std / numpy.std(vector, ddof=1))
            vector = vector + numpy.abs(numpy.mean(vector) - set_mean)
            get_pointer()[v] = vector

            with BAR_LOCK:
                BAR_VALUE.value += 1