import numpy as np

def in_bz(x, y, z, a=4.07, scale=1.0):
    l = 4 * np.pi / a * scale
    if not x + y + z <= 3 / 4 * l:
        return False
    if not x + y + z >= -3 / 4 * l:
        return False
    if not x + y - z <= 3 / 4 * l:
        return False
    if not x + y - z >= -3 / 4 * l:
        return False
    if not x - y + z <= 3 / 4 * l:
        return False
    if not x - y + z >= -3 / 4 * l:
        return False
    if not x - y - z <= 3 / 4 * l:
        return False
    if not x - y - z >= -3 / 4 * l:
        return False
    if not x <= 1 / 2 * l:
        return False
    if not x >= -1 / 2 * l:
        return False
    if not y <= 1 / 2 * l:
        return False
    if not y >= -1 / 2 * l:
        return False
    if not z <= 1 / 2 * l:
        return False
    if not z >= -1 / 2 * l:
        return False

    return True
