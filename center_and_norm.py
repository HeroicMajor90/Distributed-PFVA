#!/usr/bin/env python
from global_array import GlobalArray
import numpy as np

classMFile = "../classM.txt"
dataMFile = "../dataM.txt"

classM = GlobalArray.from_file(classMFile)
print("classM imported")
dataM = GlobalArray.from_file(dataMFile)
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

npclassM = np.loadtxt(classMFile)
npdataM = np.loadtxt(dataMFile)

npclassMean = np.mean(npclassM,axis=0)
npclassStd = np.std(npclassM,axis=0)
npclassStd = np.where(npclassStd == 0, 1, npclassStd)

npclassM = (npclassM - npclassMean)/npclassStd
npclassM = GlobalArray(npclassM)


npdataMean = np.mean(npdataM,axis=0)
npdataStd = np.std(npdataM,axis=0)
npdataStd = np.where(npdataStd == 0, 1, npdataStd)

npdataM = (npdataM - npdataMean)/npdataStd
npdataM = GlobalArray(npdataM)

classMCentFile = "../classMCent.txt"
dataMCentFile = "../dataMCent.txt"

np.savetxt(classMCentFile, classM.to_np(), delimiter=' ', fmt="%s") 
print("classM saved to " + classMCentFile)
np.savetxt(dataMCentFile, dataM.to_np(), delimiter=' ', fmt="%s") 
print("dataM saved to " + dataMCentFile)

