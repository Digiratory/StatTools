import operator
import time
from multiprocessing import Value, Lock
from threading import Thread

from numpy import ndarray, array
from collections.abc import Iterable

from rich.progress import Progress


class CheckNumpy:

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if isinstance(value, ndarray):
            instance.__dict__[self.name] = value
        elif isinstance(value, list):
            try:
                instance.__dict__[self.name] = array(value)
            except Exception:
                raise ValueError("Cannot cast input list to numpy array!")
        else:
            raise ValueError("Only list or numpy.ndarray can be used as input data!")


class CounterValue:

    def __init__(self, start: int):
        self.__counter = Value('i', start, lock=True)

    def __perform_through_type_check(self, func, other, right=False):
        if isinstance(other, int):
            if right:
                with self.__counter.get_lock():
                    return func(other, self.__counter.value)
            else:
                with self.__counter.get_lock():
                    return func(self.__counter.value, other)
        else:
            raise ValueError("Increment value is supposed to be integer!")

    @property
    def counter(self) -> int:
        return self.__counter.value

    @counter.setter
    def counter(self, val):
        if isinstance(val, type(int())):
            self.__counter.value = val
        else:
            raise ValueError("Increment value is supposed to be integer!")

    @property
    def lock(self) -> Lock:
        return self.__counter.get_lock()

    def __repr__(self):
        return f"'CV:{self.__counter.value}'"

    def __str__(self):
        return str(self.__counter.value)

    def __eq__(self, other):
        if isinstance(other, type(int())):
            self.__counter.value = other
            return self.__counter.value
        else:
            raise ValueError("Increment value is supposed to be integer!")

    def __iadd__(self, other: int):
        self.__counter.value = self.__perform_through_type_check(operator.add, other)
        return self

    def __isub__(self, other: int):
        self.__counter.value = self.__perform_through_type_check(operator.sub, other)
        return self

    def __imul__(self, other: int):
        self.__counter.value = self.__perform_through_type_check(operator.sub, other)
        return self

    def __add__(self, other: int):
        return self.__perform_through_type_check(operator.add, other)

    def __sub__(self, other: int):
        return self.__perform_through_type_check(operator.sub, other)

    def __radd__(self, other: int):
        return self.__perform_through_type_check(operator.sub, other, right=True)

    def __rsub__(self, other: int):
        return self.__perform_through_type_check(operator.sub, other, right=True)


class ProgressManager:

    def __init__(self, tasks_list, refresh_interval=0.1, elapsed_time=False):

        self.counters = [CounterValue(start=0) for i in range(len(tasks_list))]
        self.last_updates = [0 for i in range(len(tasks_list))]
        self.__stop_bit = Value('i', 0)

        self.totals = [tasks_list[task] for task in tasks_list]

        self.timeout = refresh_interval
        self.elapsed_info = elapsed_time
        Thread(target=self.updating, args=(tasks_list,)).start()

    def updating(self, tasks_list):

        with Progress() as bar:
            t1 = time.perf_counter()

            tasks = [bar.add_task(task, total=tasks_list[task]) for task in tasks_list]
            totals = [tasks_list[task] for task in tasks_list]
            total_tasks = len(tasks_list)

            while not bar.finished:

                for task_id in range(total_tasks):

                    with self.counters[task_id].lock:
                        current_diff = abs(self.counters[task_id] - self.last_updates[task_id])

                    if current_diff != 0:
                        bar.update(tasks[task_id], advance=current_diff)

                        with self.counters[task_id].lock:
                            self.last_updates[task_id] = self.counters[task_id].counter

                        if bar.tasks[task_id].completed >= totals[task_id]:
                            bar.reset(tasks[task_id])

                if self.__stop_bit.value != 0:

                    for task_id in range(total_tasks):
                        bar.tasks[task_id].completed = self.totals[task_id]

                    if self.elapsed_info:
                        print(f"[{list(tasks_list)[0]}] Elapsed: {round(time.perf_counter() - t1, 2)} seconds")

                    bar.stop()
                    return

                time.sleep(self.timeout)

    def stop(self):
        self.__stop_bit.value = 1