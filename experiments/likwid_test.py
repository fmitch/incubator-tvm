#!/usr/bin/env python

import pylikwid
import sys
import numpy as np

cpus = [0,1]
eventset = "CACHE"

arr=np.random.uniform(size=1000000)
print(arr.shape)
pylikwid.inittopology()
cputopo = pylikwid.getcputopology()
print(cputopo['activeHWThreads'])
thr = cputopo['threadPool'][0]
pylikwid.finalizetopology()

err = pylikwid.init(cpus)
if err > 0:
    print("Cannot initialize LIKWID")
    sys.exit(1)
group = pylikwid.addeventset(eventset)
if group >= 0:
    print("Eventset {} added with ID {}".format(eventset, group,))
else:
    print("Failed to add eventset {}".format(eventset))
    sys.exit(1)
err = pylikwid.setup(group)
if err < 0:
    print("Setup of group {} failed".format(group))
    sys.exit(1)
err = pylikwid.start()

s = arr.sum()
print(s)

err = pylikwid.stop()
if err < 0:
    print("Stop of group {} failed".format(group))
    sys.exit(1)
for thread in range(0,len(cpus)):
    for i in range(pylikwid.getnumberofevents(group)):
        print("Result Event {} : {}".format(pylikwid.getnameofevent(group, i), pylikwid.getresult(group,i,thread)))
pylikwid.finalize()

