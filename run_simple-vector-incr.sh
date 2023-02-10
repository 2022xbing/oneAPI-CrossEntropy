#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling CrossEntropy.cpp
dpcpp lab/CrossEntropy.cpp
if [ $? -eq 0 ]; then ./a.out; fi

