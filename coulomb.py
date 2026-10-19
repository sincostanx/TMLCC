import pandas as pd
import numpy as np
from pymatgen.io.cif import CifFile, CifParser
from pymatgen.core.structure import SiteCollection
from matminer.featurizers.structure import CoulombMatrix, BagofBonds, JarvisCFID, StructureComposition

def coulomb(file_path):
    cif_file = CifParser(file_path, occupancy_tolerance=100)
    structure = cif_file.get_structures()[0]
    featurizer = CoulombMatrix()
    featurizer.fit([structure])
    return np.array(featurizer.featurize(structure))
    
if __name__=="__main__":
    # train_size = 68613
    test_size = 2000
    max_len = 0
    for i in range(0, test_size):
        if i%10 == 0: print("now ", i, max_len)
        # file_path = "./dataset/mof_cif_train/mof_unit_" + str(i+1) + ".cif"
        file_path = "./dataset/mof_cif_pretest/mof_unit_pretest_" + str(i+1) + ".cif"
        temp = coulomb(file_path)
        np.save("./dataset/pretest_coulomb/test_coulomb_" + str(i+1) +".npy", temp)
        if temp.shape[0] > max_len: max_len = temp.shape[0]
        
    print(max_len)