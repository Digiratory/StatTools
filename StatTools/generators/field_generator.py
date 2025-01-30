# ---------------------------------------------------------------------------------------
# Генератор двумерных коррелированных полей. Python-интерпретация используемого ранее
# Mathlab-скрипта fBm2Dgen.m . На выходе получаем два поля.
#
# Как использовать:
#       в рабочем скрипте вызвать функцию start():
#           import field_generator
#           fields = field_generator.start( . . . )
#       В переменную fields записывются два массива. Чтобы получить поля:
#           field1 = fields[0]
#           field2 = fields[1]
#
# Параметры (. . .):
#       H=1.3 - целевой показатель Херста
#       quantity=1000 - количество строк в массиве поля (количество реализаций)
#       length=1440  - количество столбцов в поле (длина реализаций)
#       fast_fields=True - если True, то сам считает необходимый размер окружности по
#       ранее заданным параметрам quantity and length. По умолчанию оставить True.
#       parallel=False - распараллеливать процессы расчета или нет. В случае если
#       генерация полей производится в рамках одного процесса (до вызова генератора
#       не было попыток создать параллельные процессы) лучше указать True. При длине
#       полей 2^12 и более можно значительно сократить время расчета.
#       comments=False - выводить комментарии в консоль или нет, чтобы отслеживать
#       какой этап выполняется в данный момент.
#       progress_bar=True - выводить или нет бар прогресса (удобнее чем комментарии)
# ---------------------------------------------------------------------------------------


from multiprocessing.process import current_process
import numpy
from numpy.random.mtrand import normal
from functools import partial
from math import log, ceil
from multiprocessing import cpu_count, Pool
from multiprocessing.dummy import freeze_support
from numba import njit
from numpy import zeros, linspace, size, array, array_split, real, imag, kron, meshgrid, sqrt, gradient, savetxt, \
    zeros_like, min, max, full
from numpy.fft import fft2
from tqdm import tqdm


@njit(cache=True)
def core(x, y, R, alpha):
    if alpha <= 1.5:
        beta = 0
        c2 = alpha / 2
        c0 = 1 - alpha / 2
    else:
        beta = alpha * (2 - alpha) / (3 * R * (pow(R, 2) - 1))
        c2 = (alpha - beta * pow((R - 1), 2) * (R + 2)) / 2
        c0 = beta * pow((R - 1), 3) - c2 + 1

    r = sqrt(pow((x[0] - y[0]), 2) + pow((x[1] - y[1]), 2))
    if r <= 1:
        out = c0 - pow(r, alpha) + c2 * pow(r, 2)
    else:
        if r <= R:
            out = beta * pow((R - r), 3) / r
        else:
            out = 0

    return [out, c0, c2]


def square_array_cycle(processing_line, x_axis, y_axis, R, H, input_array):
    square = input_array
    for x in processing_line:
        for y in range(size(y_axis)):
            x1 = [x_axis[x], y_axis[y]]
            x2 = [x_axis[0], y_axis[0]]
            core_result = core(array([x_axis[x], y_axis[y]]), array([x_axis[0], y_axis[0]]), R=R, alpha=2 * H)
            square[y][x] = core_result[0]
    return square


def corner_reflector(input_array):
    if size(input_array, axis=0) != size(input_array[0], axis=0):
        raise NameError("\n-->  Corner reflector works only for square-like arrays! ")

    @njit(cache=True)
    def corner_reflector_C(input_array, zeros_array):
        output = zeros_array

        for k in range(len(input_array)):
            for k2 in range(input_array[0].size):
                output[k][k2] = input_array[k][k2]
        num = 2
        for row in range(len(input_array) - 2, 0, -1):
            row_line = input_array[row]
            output[row+num][0:len(row_line)] = input_array[row]
            num = num + 2

        num = 2
        for col in range(input_array[0].size - 2, 0, -1):
            for k in range(len(input_array)):
                output[k][col + num] = input_array[k][col]
            num = num + 2

        start_row = len(input_array) - 1
        start_col = input_array[0].size - 1
        length = len(input_array)
        indeces = [length, length]
        for r in range(start_row - 1, 0, -1):
            for c in range(start_col - 1, 0, -1):
                output[indeces[0], indeces[1]] = input_array[r][c]
                indeces[1] += 1
            indeces[0] += 1
            indeces[1] = length

        return output

    x = array([[1, 2, 3, 44], [4, 5, 6, 77], [7, 8, 9, 99], [10, 11, 12, 13]], dtype=float)
    big = 2 * size(x, axis=0) - 2
    zeros_array = zeros((big, big), dtype=float)
    compile_C = corner_reflector_C(x, zeros_array)

    big = 2 * size(input_array, axis=0) - 2
    zeros_array = zeros((big, big), dtype=float)
    result = corner_reflector_C(input_array, zeros_array)

    return result

@njit(cache=True)
def crop_out(input_array, cropped, given_length, given_width):

    for k in range(given_length):
        for k2 in range(given_width):
            cropped[k][k2] = input_array[k][k2]

    return cropped

def fields(H, length, comments=False, R=2, parallel=False, return_axises=False, progress_bar=False):
    if progress_bar:
        if not parallel:
            try:
                current = current_process()
                proc_num = current._identity[0] - int(cpu_count() / 2)
                string = f"[P:{proc_num}]Generating fields: "
            except:
                string = f"Generating fields: "
        else:
            string = f"Generating fields: "

        bar = tqdm(total=6, desc=string, leave=True)
    power_of_2 = log(length, 2)
    decimal = float(str(power_of_2 - int(power_of_2))[1:])
    if decimal == 0:
        grid_size = length
    else:
        grid_size = pow(2, ceil(log(length, 2)))
        if comments:
            print(f"Grid size is going to be bigger than input length: {grid_size}")

    width = grid_size
    x_axis = linspace(1, length, length) * R / length
    y_axis = linspace(1, width, width) * R / width
    square = zeros((length, width), dtype=float)

    if parallel:
        cpu_to_use = int(cpu_count() / 2)
        line = array_split(linspace(0, size(x_axis) - 1, size(x_axis), dtype=int), 6)
        if comments:
            print("Started parallel processing . . .")
        pool = Pool(processes=cpu_to_use)
        result = pool.map(partial(square_array_cycle, y_axis=y_axis, R=R, H=H, input_array=square, x_axis=x_axis), line)
        pool.terminate()

        num_part = 0
        for p in result:
            part = p

            @njit(cache=True)
            def extracting_result(square, part, width, line):
                for k in line:
                    for row in range(width):
                        square[row][k] = part[row][k]
                return square

            square = extracting_result(square, part, width, line[num_part])
            num_part += 1

    else:
        line = linspace(0, size(x_axis) - 1, size(x_axis), dtype=int)
        if comments:
            print("Started processing . . .")
        square = square_array_cycle(line, x_axis, y_axis, R, H, square)

    if progress_bar:
        bar.update()

    if comments:
        print("Started corner reflecting  . . .")
    corner_reflected = corner_reflector(square)
    if progress_bar:
        bar.update()
    square = None
    if comments:
        print("LAM processing has been started . . . ")

    numpy.seterr(all='raise')
    savetxt('h_1_zeros.txt', corner_reflected)
    lam = real(fft2(corner_reflected)) / (4 * (width - 1) * (length - 1))
    try:
        lam = sqrt(lam)
    except:
        if max(corner_reflected) < 0.00001 and abs(min(corner_reflected)) < 0.00001:
            lam = zeros_like(corner_reflected)
        else:
            raise NameError("\n--> LAM problem: max and min are more substantial than 0.00001 . . .")
    if progress_bar:
        bar.update()

    corner_reflected = None

    @njit(cache=True)
    def complex_multiplier(input_array, real_random_array, image_random_array, empty_array):

        for k1 in range(len(empty_array)):
            for k2 in range(empty_array[0].size):
                empty_array[k1][k2] = complex(real_random_array[k1][k2], image_random_array[k1][k2])

        return input_array * empty_array

    real_random = normal(0, 1, size=[size(lam, axis=0), size(lam[0], axis=0)])
    image_random = normal(0, 1, size=[size(lam, axis=0), size(lam[0], axis=0)])
    empty_array = zeros((size(lam, axis=0), size(lam[0], axis=0)), dtype=complex)

    if comments:
        print("Now doing complex multiplying  . . .")
    if progress_bar:
        bar.update()

    F = complex_multiplier(lam, real_random, image_random, empty_array)
    lam = None
    real_random = None
    image_random = None
    empty_array = None
    F = fft2(F)

    if comments:
        print("Cropping out matrix of given size . . .")
    if progress_bar:
        bar.update()
    F = crop_out(F, zeros((length, width), dtype=complex), length, width)

    core_result = core(array([0, 0]), array([0, 0]), R=2, alpha=2*H)
    c2 = core_result[2]

    field1 = real(F)
    field2 = imag(F)
    F = None

    field1 = field1 - field1[0][0]
    field2 = field2 - field2[0][0]

    y_new = y_axis * normal(0, 1, 1)
    y_new = array([y_new], dtype=float)
    y_new = y_new.T

    x_new = x_axis * normal(0, 1, 1)

    field1 = field1 + kron(y_new, x_new) * sqrt(2 * c2)
    field2 = field2 + kron(y_new, x_new) * sqrt(2 * c2)

    X, Y = meshgrid(x_axis, y_axis)


    """def cut_out_the_disk(input_array, grid, nan_count=False):
        field = input_array
        nan_dots = 0
        for x in range(len(grid)):
            for y in range(grid[0].size):
                if grid[x][y] > 1:
                    field[x][y] = None
                    if nan_count:
                        nan_dots += 1
        if nan_count:
            return field, nan_dots
        else:
            return field

    if comments:
        print("Only thing left is to cut out unit disk . . .")"""

    @njit(cache=True)
    def cut_out_the_disk(input_array, grid, nan_array):
        field = input_array
        for x in range(len(grid)):
            for y in range(grid[0].size):
                if grid[x][y] <= 1:
                    nan_array[x][y] = field[x][y]
        return nan_array

    def compile_C():
        a = full((4, 4), 1, dtype=float)
        X = array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=float)
        Y = array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=float)
        X, Y = meshgrid(X, Y)
        grid = X * X + Y * Y
        nan = full((size(a, axis=0), size(a, axis=0)), None, dtype=float)
        test = cut_out_the_disk(a, grid, nan)

    compile_C()

    grid = X * X + Y * Y
    nan_array = full((size(field1, axis=0), size(field1, axis=0)), None, dtype=float)
    field1 = cut_out_the_disk(field1, grid, nan_array)
    field2 = cut_out_the_disk(field2, grid, nan_array)
    grid = None
    field1 = crop_out(field1, zeros((int(length/2), int(width/2))), int(length / 2), int(width / 2))
    field2 = crop_out(field2, zeros((int(length / 2), int(width / 2))), int(length / 2), int(width / 2))
    X = crop_out(X, zeros((int(length / 2), int(width / 2))), int(length / 2), int(width / 2))
    Y = crop_out(Y, zeros((int(length / 2), int(width / 2))), int(length / 2), int(width / 2))
    if progress_bar:
        bar.update()

    if comments:
        print("-> Returning result . . .")
    if return_axises:
        return [field1, field2, X, Y]
    else:
        return [field1, field2]

def start(H, quantity, length, fields_length=0, parallel=False, comments=False, R=2,
          fast_fields=True, return_axises=False, progress_bar=False):

    def parameters_handler(quantity, length, fields_length, comments, fast_fields):
        sufficient_field_size = int(sqrt(pow(quantity, 2) + pow(length, 2))) * 2
        log_found = pow(2, ceil(log(sufficient_field_size, 2)))

        if fast_fields:
            if comments:
                print(f"Field's sizes are going to be {log_found} x {log_found}")
            return log_found

        if fields_length != 0:
            try:
                max_num_real = int(sqrt(pow((fields_length / 2), 2) - pow(length, 2)))
            except:
                raise NameError(f"\n---> Field size is way smaller than given length . . .")

            if max_num_real < quantity:
                print("With given quantity and length, algo wouldn't be able to extract your matrix out of the unit disk . . ."
                      f"\nMax number of realizations with length of {length} is {max_num_real}")
                raise NameError("\nWith given quantity and length, algo wouldn't be able to extract your matrix out of the unit disk . . ."
                                f"\nMax number of realizations with length of {length} is {max_num_real}"
                                f"\n---> Field is not big enough to fit desired group of realizations!")
            else:

                if comments and not fast_fields:
                    if sufficient_field_size < fields_length:
                        print(f"->You can use smaller field to find your set of realizations: field_length = {sufficient_field_size} is enough"
                              f"\n->{sufficient_field_size} transforms into : 2^(ceil(log({sufficient_field_size}, 2))) = {log_found}"
                              f"\n->Use field_generator.start( . . . , fast_fields=True) to perform operations faster.")

                return fields_length
        else:
            if fast_fields:
                if comments:
                    print(f"Field's sizes are going to be {log_found} x {log_found}")
                return log_found
            else:
                raise NameError(f"\n---> You need to either set size of fields or to use fast_fields=True!")

    fields_length = parameters_handler(quantity, length, fields_length, comments, fast_fields)

    if H > 1:
        res = fields(H=H - 1, length=fields_length, comments=comments, R=R, parallel=parallel,
                     return_axises=return_axises, progress_bar=progress_bar)
    else:
        res = fields(H=H, length=fields_length, comments=comments, R=R, parallel=parallel,
                     return_axises=return_axises, progress_bar=progress_bar)
    field1 = crop_out(res[0], zeros((quantity, length), dtype=float), quantity, length)
    field2 = crop_out(res[1], zeros((quantity, length), dtype=float), quantity, length)

    if H <= 1:
        grad1 = gradient(field1)
        grad2 = gradient(field2)
        field1 = grad1[1]
        field2 = grad2[1]
        grad1 = None
        grad2 = None

    if return_axises:
        X = crop_out(res[2], zeros((quantity, length), dtype=float), quantity, length)
        Y = crop_out(res[3], zeros((quantity, length), dtype=float), quantity, length)

    res = None

    if return_axises:
        return [field1, field2, X, Y]
    else:
        return [field1, field2]

if __name__ == '__main__':
    freeze_support()
    print("Use field_generator.start() for full processing . . .")
    ret = start(1.3, 1000, 1440, parallel=True, comments=True, return_axises=True)

    print(ret)
    print(ret[0].shape)