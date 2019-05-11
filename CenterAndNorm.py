#!/usr/bin/env python
from global_array import GlobalArray
import numpy as np

classM = GlobalArray.from_file("classM.txt")
print("classM imported")
dataM = GlobalArray.from_file("dataM.txt")
print("dataM imported")

classMean = classM.mean(axis=0)
print("class mean calculated")
classStd = classM.std(axis=0,zero_default=1)
print("class std calculated")

classMean = classMean.transpose()
classStd = classStd.transpose()

classM = (classM - classMean)/classStd
print("classM centered and normalized")

dataMean = dataM.mean(axis=0)
print("data mean calculated")
dataStd = dataM.std(axis=0,zero_default=1)
print("data std calculated")

dataMean = dataMean.transpose()
dataStd = dataStd.transpose()

dataM = (dataM - dataMean)/dataStd
print("dataM centered and normalized")

classMCentFile = "classMCent.txt"
dataMCentFile = "dataMCent.txt"

np.savetxt(classMCentFile, classM.to_np(), delimiter=' ', fmt="%s") 
print("classM saved to " + classMCentFile)
np.savetxt(dataMCentFile, dataM.to_np(), delimiter=' ', fmt="%s") 
print("dataM saved to " + dataMCentFile)

