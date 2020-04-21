import numpy as  np
cimport numpy as np
import array
import bisect


cpdef float sorted_median(float[:] data, int i, int j):
    cdef int n = j - i
    cdef int mid
    if n == 0:
        raise Exception("no median for empty data")
    if n % 2 == 1:
        return data[i + n // 2]
    else:
        mid = i + n // 2
        return (data[mid - 1] + data[mid])/2

cpdef median_filter(np.ndarray data, int window, bint need_two_end=False):
    cdef int w_len = window // 2 * 2 + 1
    cdef int t_len = len(data)
    cdef float[:] val = array.array('f', [x for x in data])
    cdef float[:] ans = array.array('f', [x for x in data])
    cdef float[:] cur_windows = array.array('f', [0 for x in range(w_len)])
    cdef int delete_id
    cdef int add_id
    cdef int index
    if t_len < w_len:
        return ans
    for i in range(0, w_len):
        index = i
        add_id = bisect.bisect_right(cur_windows[:i], val[i])
        while index > add_id:
            cur_windows[index] = cur_windows[index - 1]
            index -= 1
        cur_windows[add_id] = data[i]
        if i >= w_len // 2 and need_two_end:
            ans[i - w_len // 2] = sorted_median(cur_windows, 0, i + 1)
    ans[window // 2] = sorted_median(cur_windows, 0, w_len)
    for i in range(window // 2 + 1, t_len - window // 2):
        delete_id = bisect.bisect_right(cur_windows, val[i - window // 2 - 1]) - 1
        index = delete_id
        while index < w_len - 1:
            cur_windows[index] = cur_windows[index + 1]
            index += 1

        add_id = bisect.bisect_right(cur_windows[:w_len - 1], val[i + window // 2])
        index = w_len - 1
        while index > add_id:
            cur_windows[index] = cur_windows[index - 1]
            index -= 1
        cur_windows[add_id] = data[i + window // 2]

        ans[i] = sorted_median(cur_windows, 0, w_len)

    if need_two_end:
        for i in range(t_len - window // 2, t_len):
            delete_id = bisect.bisect_right(cur_windows[: w_len], data[i - window // 2 - 1]) - 1
            index = delete_id
            while index < w_len - 1:
                cur_windows[index] = cur_windows[index + 1]
                index += 1
            w_len -= 1
            ans[i] = sorted_median(cur_windows[: w_len], 0, w_len)

    return ans
