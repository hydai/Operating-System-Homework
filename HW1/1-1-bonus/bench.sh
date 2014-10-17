#!/bin/sh
make
./myfork ./testcase/Normal ./testcase/abort ./testcase/alarm ./testcase/segmentFault
make clean
