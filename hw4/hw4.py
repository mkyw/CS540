import csv
from dis import dis
import numpy as np
import math

def load_data(filepath):
    list = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        categories = ("HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed")
        for row in reader:
            filtered_d = dict((k, row[k]) for k in categories)
            list.append(filtered_d)
    return list

def calc_features(row):
    x1 = int(row.get('Attack'))
    x2 = int(row['Sp. Atk'])
    x3 = int(row['Speed'])
    x4 = int(row['Defense'])
    x5 = int(row['Sp. Def'])
    x6 = int(row['HP'])
    a = np.array([x1, x2, x3, x4, x5, x6], dtype=np.int64)
    return a

def hac(features):
    n = len(features)
    distArray = np.zeros([n, n], dtype=float)
    numPokemons = np.zeros([n], dtype=int)
    for i in range(len(distArray)):
        numPokemons[i] = 1
        for j in range(len(distArray[i])):
            distArray[i][j] = math.dist(features[i], features[j])
            print(distArray[i][j])

    # d(A, B) = max(math.dist(a,b))
    Z = np.empty((n-1,4), dtype=float)

    index = [i for i in range(n)]

    for x in range(len(Z)):
        c1 = -1
        c2 = -1
        dist = math.inf

        for y in range(len(index)):
            for z in range(len(index[y])):
                if ( (index[y]!=index[z])):
                    if ((distArray[index[y]][index[z]] < dist)):
                        c1 = index[y]
                        c2 = index[z]
                        dist = distArray[index[y]][index[z]]
                index.remove[y]
                index.remove[z]

        Z[x][0] = c1
        Z[x][1] = c2
        Z[x][2] = dist
        Z[x][3] = numPokemons[c1] + numPokemons[c2]

        # add new column to distance array
        distArray
    return Z

#def imshow_hac(Z):
