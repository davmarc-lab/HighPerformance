#!/bin/sh
NUM=10

if [[ $# -eq 0 ]] ; then
    FILE="serial.txt"
    EXEC="./circles"
fi

if [[ $# -eq 1 ]] ; then
    FILE="serial.txt"
    EXEC="./circles"
    NUM=$1
fi

if [[ $# -eq 2 ]] ; then
    FILE="openmp.txt"
    EXEC="circles.omp"
    NUM=$2
fi
if [[ ! -e $FILE ]] ; then 
    touch $FILE  
fi

# Executing the file
for (( i=0; i<$NUM; i++ ))
do  
    ((num=$i+1))
    echo "Iteration number $num"
    echo "Executing $EXEC file ..."
    output="`./$EXEC`"
    lastword=""

    # find the last word and copy the text
    # Could be better!?
    for line in $output ; do
        lastword=$line
    done

    # Write in the chosen file the results
    echo "Elapsed time: $lastword s" >> $FILE

    echo "" >> $FILE
    echo "Result appended to $FILE"   
done



