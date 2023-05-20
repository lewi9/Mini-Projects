import numpy as np
import matplotlib.pyplot as plt


def init_map(rows, cols):
    map = np.zeros((rows, cols))
    for i in range(map.shape[0]):
        counter = 0
        type = 0
        if np.random.randint(2):
            type = 3
        for j in range(map.shape[1]):
            if counter == 0:
                if type == 3:
                    type = 0
                    counter = np.random.randint(4, 8)
                else:
                    type = 3
                    counter = np.random.randint(1, 4)

            map[i][j] = type
            counter -= 1
    return map


def roll_map(map):
    for i in range(map.shape[0]):
        if i % 2:
            map[i] = np.roll(map[i], 1)
        else:
            map[i] = np.roll(map[i], -1)
    return map


def show_map(map):
    row = [4 for _ in range(map.shape[1])]
    for j in range(3):
        map = np.vstack((row, map))
        map = np.vstack((map, row))
    itemindex = np.where(map == 2)
    if itemindex[0].size == 0:
        map[2][map.shape[1] // 2] = 2

    x = []
    y = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            for k in range(int(map[i][j])):
                x.append(j)
                y.append(i)

    xedges = np.arange(0, map.shape[1] + 1, 1)
    yedges = np.arange(0, map.shape[0] + 1, 1)

    return np.histogram2d(x, y, bins=(xedges, yedges))


def create_frame(map, t):
    H, xedges, yedges = show_map(map)
    plt.imshow(H.T, interpolation='nearest', origin='lower', cmap="coolwarm",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.savefig(f'./frames/frame_{t}.png', transparent=False, facecolor='white')
    plt.close()


def move(map):
    itemindex = np.where(map == 2)
    if itemindex[0].size != 0:
        try:
            if map[itemindex[0][0] + 1][itemindex[1][0]] == 0:
                return np.array([])
        except:
            return np.array(["win"])

        map[itemindex[0][0]][itemindex[1][0]] = 3
        try:
            map[itemindex[0][0] + 1][itemindex[1][0]] = 2
        except:
            pass
    else:
        if map[0][map.shape[1] // 2] == 0:
            return np.array([])
        map[0][map.shape[1] // 2] = 2
    return map


def cut_map(map, rows, cols):
    itemindex = np.where(map == 2)
    if itemindex[0].size == 0:
        upper_row = map[0, map.shape[1] // 2 - cols // 2:map.shape[1] // 2 + cols // 2 + cols % 2]
        middle_row = np.array([0 for i in range(cols)])
        middle_row[middle_row.shape[0] // 2] = 2
        lower_row = np.array([0 for i in range(cols)])
        return np.vstack((lower_row, middle_row, upper_row))
    else:
        yd = itemindex[0][0] - rows // 2
        yu = itemindex[0][0] + rows // 2 + rows % 2
        xl = itemindex[1][0] - cols // 2
        xr = itemindex[1][0] + cols // 2 + cols % 2
        xn_l = xl
        xn_r = xr
        yn_d = yd
        yn_u = yu
        if xl < 0:
            xn_l = 0
        if xr > map.shape[1]:
            xn_r = map.shape[1]
        if yd < 0:
            yn_d = 0
        if yu > map.shape[0]:
            yn_u = map.shape[0]
        segment = map[yn_d:yn_u, xn_l:xn_r]

        meta = [3 for i in range(cols)]
        while xn_l > xl:
            xn_l -= 1
            zeros_h = [[0] for i in range(segment.shape[0])]
            segment = np.hstack((zeros_h, segment))
        while xn_r < xr:
            xn_r += 1
            zeros_h = [[0] for i in range(segment.shape[0])]
            segment = np.hstack((segment, zeros_h))
        while yn_d > yd:
            yn_d -= 1
            zeros_v = [0 for i in range(segment.shape[1])]
            segment = np.vstack((zeros_v, segment))
        while yn_u < yu:
            yn_u += 1
            meta = [3 for i in range(segment.shape[1])]
            segment = np.vstack((segment, meta))

        return segment
