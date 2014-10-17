#!/bin/sh
make
./UserModeMonitor ./testcase/Normal
echo " "
./UserModeMonitor ./testcase/alarm
echo " "
./UserModeMonitor ./testcase/abort
echo " "
./UserModeMonitor ./testcase/segmentFault
make clean
