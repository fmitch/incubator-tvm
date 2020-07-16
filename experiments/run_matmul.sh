#!/bin/bash

#for BENCHMARK in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ; 
#BENCHMARK=7
#FEATURE=itervar
#SIZE=wide
#LOGFILE="$FEATURE"_"$SIZE"_benchmark_"$BENCHMARK".out
#python3 -u benchmarks_conv2d.py $BENCHMARK $FEATURE $SIZE 5  | tee $LOGFILE
#FEATURE=datavol_repeat
#SIZE=small
#LOGFILE="$FEATURE"_"$SIZE"_benchmark_"$BENCHMARK".out
#python3 -u benchmarks_conv2d.py $BENCHMARK $FEATURE $SIZE 5  | tee $LOGFILE
#FEATURE=datavol_repeat
#SIZE=wide
#LOGFILE="$FEATURE"_"$SIZE"_benchmark_"$BENCHMARK".out
#python3 -u benchmarks_conv2d.py $BENCHMARK $FEATURE $SIZE 5  | tee $LOGFILE

#for BENCHMARK in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ; 


for BENCHMARK in 0 1 2 3 4 5  ; 
do
  for FEATURE in itervar_silent_dv datavol time ;
  do
    LOGFILE=matmul_"$FEATURE"_benchmark_"$BENCHMARK"_1core_"$1".out
    echo $LOGFILE
    python3 -u benchmarks_matmul.py -b $BENCHMARK -f $FEATURE -n 5 -t 1000  | tee $LOGFILE
  done
done
