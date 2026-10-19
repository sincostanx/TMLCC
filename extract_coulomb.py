import numpy as np
import pandas as pd

if __name__=="__main__":
    # train_size = 68613
    test_size = 2000
    max_len = 1584 # for train set 1296 for test
    
    feature = np.zeros((test_size, max_len+1))
    for i in range(test_size):
        print("now ", i)
        file_path = "./dataset/pretest_coulomb/test_coulomb_" + str(i+1) + ".npy"
        # file_path = "./dataset/pretest_coulomb/train_coulomb_" + str(i+1) +".npy"
        temp = np.load(file_path)
        feature[i][0] = temp.shape[0]
        feature[i][1:temp.shape[0]+1] = temp 
        
    df = pd.DataFrame(feature)
    df.to_csv("pretest-coulomb.csv", index = False)