
import numpy as np
import scipy.linalg
import pprint
pp = pprint.PrettyPrinter(indent=4)

os = [-1, 0, +1]

cs = []
for o1 in os:
    for o2 in os:
        for o3 in os:
            for o4 in os:
                cs.append((o1, o2, o3, o4))

# Remove options without any alignment or rebellion
for c in cs[:]:
    if -1 not in c or +1 not in c: 
        cs.remove(c)
# Remove options with mutual rebellion
for c in cs[:]:
    if (-1 in (c[0], c[3])) or (c[1] == -1 and c[2] == -1):
        cs.remove(c)
# Remove symmetric duplicates
for c in cs[:]:
    if c[::-1] in cs:
        cs.remove(c)
# Remove reducible states (those with one particle non-interacting)
for c in cs[:]:
    if (c[0] == 0 and c[1] == 0) or (c[2] == 0 and c[3] == 0):
        cs.remove(c)

print('There are {} variants:'.format(len(cs)))
for c in cs:
    for o in c:
        print(str(o) if o != -1 else 'm', end='')
    m = np.array(c).reshape([2, 2])
    # print(c)
    print()
    print(scipy.linalg.eig(m)[0])
