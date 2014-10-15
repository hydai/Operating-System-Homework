#!/bin/sh
make
./UserModeMonitor ./mac/p1
echo " "
./UserModeMonitor ./mac/p2
echo " "
./UserModeMonitor ./mac/p3
echo " "
./UserModeMonitor ./mac/p4
make clean
