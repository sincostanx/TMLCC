import numpy as np
import pandas as pd
from zipfile import ZipFile
pred = np.load('/export/work/worameth/TMLCC/pred-test/baseline0.npy')

answer = pd.DataFrame({
    "id": ["pretest_" + str(i) for i in range(1,2001)],
    "CO2_working_capacity [mL/g]": pred.T[0]
    })

answer.to_csv("submission.csv", index=False)

submission = ZipFile("submission.zip", "w")
submission.write("submission.csv")
submission.close()