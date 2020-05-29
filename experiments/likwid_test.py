#!/usr/bin/env python

import pylikwid
import sys

cpus = [0]
eventset = "RETIRED_INSTRUCTIONS:PMC0,ICACHE_FETCHES:PMC1"

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
if err < 0:
    print("Start of group {} failed".format(group))
    sys.exit(1)

arr=[]
for i in range(500000000):
    arr.append(i)
err = pylikwid.stop()
if err < 0:
    print("Stop of group {} failed".format(group))
    sys.exit(1)
for thread in range(0,len(cpus)):
    print("Result CPU {} : {}".format(cpus[thread], pylikwid.getresult(group,0,thread)))
    print("Result CPU {} : {}".format(cpus[thread], pylikwid.getresult(group,1,thread)))
pylikwid.finalize()

