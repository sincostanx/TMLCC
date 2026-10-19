import numpy as np
import pandas as pd

# fragments = ["0-14999",
#             "15000-22974",
#             "22949-29999",
#             "30000-44999",
#             "45000-59999",
#             "60000-68612",
#             ]

# train = pd.read_csv("./dataset/train.csv")
# gsa = np.zeros((len(train.index), 2))

# print(gsa.shape)
# fragment = np.array([np.load("./dataset/surface-fragment/" + file + ".npy") for file in fragments])

# for i in range(len(train.index)):
#     if train['surface_area [m^2/g]'].iloc[i] <= 0: gsa[i][1] = 1
#     # print(fragment.shape)
#     gsa[i][0] = np.amax(fragment[:, i])
#     if gsa[i][0] == 0: print("Error at ", i)

# surface = pd.DataFrame(gsa)
# surface.to_csv("train-surface.csv", index=False)


train = pd.read_csv("./dataset/pretest.csv")
gsa = np.zeros((len(train.index), 2))

print(gsa.shape)
fragment = np.array([np.load("./dataset/pretest-gsa_array.npy")])

for i in range(len(train.index)):
    if train['surface_area [m^2/g]'].iloc[i] <= 0: gsa[i][1] = 1
    print(train['surface_area [m^2/g]'].iloc[i])
    # print(fragment.shape)
    gsa[i][0] = np.amax(fragment[:, i])
    if gsa[i][0] == 0: print("Error at ", i)

# surface = pd.DataFrame(gsa)
# surface.to_csv("pretest-surface.csv", index=False)
    
# test = pd.read_csv("./dataset/train.csv")
# canFix = []
# l = list(test[test.iloc[:, 5]==0].index)
# r = list(test[test.iloc[:, 4]==0].index)
# for x in l:
#     if x not in r: canFix.append(x)

# for x in r:
#     if x not in l:
#         if x not in canFix: canFix.append(x)
# print(canFix, len(canFix))
# test = test['surface_area [m^2/g]']
# temp = np.zeros(len(test.index))
# test = pd.concat([test, pd.DataFrame(temp)], axis=1)
# test.to_csv("./dataset/pretest-surface.csv", index=False)