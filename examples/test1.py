import numpy as np
# a = [[[1,1], [1,1], [1,1], [2,2], [2,2], [3,3]], [[3,3], [3,3], [2,2], [2,2], [4,4], [5,5]]]

a = [[[1,1],[3,3]],
     [[1,1],[3,3]],
     [[1,1],[2,2]],
     [[2,2],[2,2]],
     [[2,2],[4,4]],
     [[3,3],[5,5]]]
a = np.array(a)
b = []
for i in range(len(a[0])):
    b_i = []
    idx = 0
    while (idx < len(a)):
        create_new = False

        if (idx == 0):
            create_new = True
        else:
            diff = a[idx][i] - a[idx-1][i]
            print('a[idx][i]: ', a[idx-1][i])
            print('a[idx][i-1]: ', a[idx-1][i])
            print('diff: ', diff)

            if np.linalg.norm(diff) == 0:
                create_new = False
            else:
                create_new = True
            print('create_new: ', create_new)

        if create_new:
            b_i_j = []
            b_i.append(b_i_j)

        position = []
        for k in range(len(a[idx][i])):
            position.append(a[idx][i][k])

        b_i[-1].append(position)
        idx += 1
    b.append(b_i)

print(b)