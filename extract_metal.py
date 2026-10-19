import pandas as pd
import numpy as np
from pymatgen.io.cif import CifFile, CifParser
from pymatgen.io.xyz import XYZ
from pymatgen.core.structure import SiteCollection
import os

train_size = 2000
s = [[] for _ in range(train_size)]
for i in range(train_size):
    if i%100 == 0: print("now ", i)
    file_path = "./dataset/mof_cif_pretest/mof_unit_pretest_" + str(i+1) + ".cif"
    cif_file = CifParser(file_path, occupancy_tolerance=100)
    structure = cif_file.get_structures()[0]
    for j in range(len(structure.sites)):
        if structure.sites[j].species.elements[0].is_metal and structure.sites[j].species.elements[0].number not in s[i]:
            # print(i, structure.sites[j].species.elements[0])
            s[i].append(structure.sites[j].species.elements[0].number)
        if len(s[i]) > 1: print("more than one metal at ", i)
        # print(structure.sites[j].species.elements[0].is_metal)
        #print(type(structure.sites[0].species.elements[0].is_metal))
    # print(s[i][0])
    np.save("./dataset/pretest-metal/test_metal_" + str(i+1) +".npy", np.array(s[i]))