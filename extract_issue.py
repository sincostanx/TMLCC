import pandas as pd
import numpy as np

df = pd.read_csv("./dataset/pretest.csv")
trainsize = 2000
l = []
for i in range(trainsize):
    if (df['surface_area [m^2/g]'].iloc[i] <= 0):
        # if (df['CO2/N2_selectivity'].iloc[i] <= 0) or (df['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'].iloc[i] == np.inf) or np.isnan(df['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'].iloc[i]): continue
        # else: l.append(i+1)
        l.append(i+1)

df2 = pd.DataFrame(np.array(l))
df2.to_csv("missing-actual-pretest.csv", index =False)