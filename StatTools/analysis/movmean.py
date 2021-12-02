from numpy import convolve, ones, mean, append, insert, ndarray, array


def movmean(arr: ndarray, k):
    if k > len(arr):
        raise ValueError("Window size > len(input_array)!")
    conv = convolve(arr, ones((k,)) / k, mode='valid')
    if k == len(conv):
        return conv
    i = k // 2 + 1
    l = len(arr)
    while True:
        conv = insert(conv, 0, mean(arr[0:i]))
        if len(conv) == l: break
        conv = append(conv, mean(arr[-i:]))
        if len(conv) == l: break
        i -= 1
    return conv


if __name__ == '__main__':
    A = array([4, 8, 6, -1, -2, -3, -1, 3, 4, 5])
    # A = normal(10, 3, 1000)
    m = movmean(A, 11)
    print(m, len(m))