#!/bin/sh
#####################################################################
# go.sh
# Generated by bench-run.pl
# /home/students/ruka7894/benchmarks/bin/bench-run.pl --bench spec-cpu2006:int:403.gcc:train --build base --prefix pin -t /home/students/ruka7894/Checkpoint1A/obj-intel64/checkpoint1A.so -o out_403_gcc.out -l 2000000 -- --copy output.out --log LOG
#####################################################################

#####################################################################
# Creating logfile
date >> LOG
#####################################################################

#####################################################################
# benchmark: spec-cpu2006/int/403.gcc input: train 0
#####################################################################
# log file update
echo -n "Started : spec-cpu2006/int/403.gcc input: train 0 at " >> LOG 
date >> LOG
# setup
ln -s /home/students/ruka7894/benchmarks/src/spec-cpu2006/int/403.gcc/builds/base/403.gcc /home/students/ruka7894/Checkpoint1A/403.gcc
ln -s -f -n --target-directory=/home/students/ruka7894/Checkpoint1A /home/students/ruka7894/benchmarks/src/spec-cpu2006/int/403.gcc/data/train/input/integrate.i
# setup experiment files
# run benchmark
pin -t /home/students/ruka7894/Checkpoint1A/obj-intel64/checkpoint1A.so -o out_403_gcc.out -l 2000000 -- /home/students/ruka7894/Checkpoint1A/403.gcc integrate.i -o integrate.s > integrate.out 2> integrate.err 
# verify benchmark
# post-copy benchmark
cp /home/students/ruka7894/Checkpoint1A/output.out /home/students/ruka7894/benchmarks/src/spec-cpu2006/int/403.gcc/builds/base/output.out;
# run custom script
# clean benchmark
rm -f /home/students/ruka7894/Checkpoint1A/403.gcc
rm -r -f integrate.i integrate.err integrate.out integrate.s
# clean experiment files
# log file update
echo -n "Finished: spec-cpu2006/int/403.gcc input: train 0 at " >> LOG 
date >> LOG
#####################################################################
# Finishing logfile
date >> LOG
#####################################################################

