

def build_list_from_input(str_in: str):
    """
    Takes a string formatted like 0-2,3,5,7,9-12 and parses into a list containing all numbers,
    with endpoints of ranges included.

    Parameters
    ----------
    str_in: str
        Numbers to include. Individual numbers separated by commas are included, as well as ranges like 6-10
        which will include numbers 6, 7, 8, 9, and 10.

    Returns
    -------
    list[int]
        List of integers. For the example in the docstring, list would include [0, 1, 2, 3, 5, 7, 9, 10, 11, 12].
    """
    nums = []
    entries = str_in.split(',')
    for entry in entries:
        if '-' in entry:
            small_num, big_num = entry.split('-')
            nums.extend(range(int(small_num), int(big_num) + 1))
        else:
            nums.append(int(entry))

    return nums
