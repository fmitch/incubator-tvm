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


for BENCHMARK in 0 2 4 6 8 10 12 14 16 18 ; 
do
  for FEATURE in itervar_silent_dv datavol ;
  do
    LOGFILE=fix_1000_"$FEATURE"_huge_benchmark_"$BENCHMARK"_1core_"$1".out
    echo $LOGFILE
    #python3 -u benchmarks_conv2d.py -b $BENCHMARK -f $FEATURE -s huge -n 10 -t 1000 -k $1  | tee $LOGFILE
    python3 -u benchmarks_conv2d.py -b $BENCHMARK -f $FEATURE -s huge -n 5 -t 1000  | tee $LOGFILE
  done

  for PREDICTION in time ;
  do
    LOGFILE=fix_"$PREDICTION"_huge_1000_benchmark_"$BENCHMARK"_1core_"$1".out
    echo $LOGFILE
    #python3 -u benchmarks_dvmodel_conv2d.py -b $BENCHMARK -p $PREDICTION -s huge -n 10 -t 1000 -k $1 | tee $LOGFILE
    python3 -u benchmarks_dvmodel_conv2d.py -b $BENCHMARK -p $PREDICTION -s huge -n 5 -t 1000 | tee $LOGFILE
    killall python3
    sleep 4
  done
done
