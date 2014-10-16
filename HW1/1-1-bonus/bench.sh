#!/bin/sh
cd ../1-1-UserModeMonitor/
make
mv UserModeMonitor ../1-1-bonus/
make clean
cd -
make
./myfork ./testcase/Normal ./testcase/abort ./testcase/alarm ./testcase/segmentFault
make clean
