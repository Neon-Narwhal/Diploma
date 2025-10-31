def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1