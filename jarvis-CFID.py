import pandas as pd
import numpy as np
from pymatgen.io.cif import CifFile, CifParser
from pymatgen.core.structure import SiteCollection
from matminer.featurizers.structure import JarvisCFID
import time

def jarvis(file_path):
    cif_file = CifParser(file_path, occupancy_tolerance=100)
    structure = cif_file.get_structures()[0]
    featurizer = JarvisCFID()
    featurizer.fit([structure])
    return np.array(featurizer.featurize(structure))
    
if __name__=="__main__":
    train_size = 68613
    # test_size = 2000
    max_len = 0
    for i in range(train_size-1, -1, -1):
        start_time = time.time()
        file_path = "./dataset/mof_cif_train/mof_unit_" + str(i+1) + ".cif"
        # file_path = "./dataset/mof_cif_pretest/mof_unit_pretest_" + str(i+1) + ".cif"
        temp = jarvis(file_path)
        np.save("./dataset/train-jarvis/train_jarvis_" + str(i+1) +".npy", temp)
        if temp.shape[0] > max_len: max_len = temp.shape[0]
        if i%1 == 0: print("now ", i, max_len, time.time() - start_time)
        
    print(max_len)