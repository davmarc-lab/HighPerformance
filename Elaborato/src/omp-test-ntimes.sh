#/bin/bash

file=$1

ot="omp-out"
if [ ! -e $ot ] ; then
    mkdir $ot
fi

make omp

rs="omp-result-ntimes.txt"
if [ -e $rs ] ; then
    rm $rs
fi
touch $rs

echo $file >> $rs

n=5

for i in {1..5} ;
do
    ./omp-skyline < $file > $ot/$ot.txt
    head "$ot/$ot.txt" -n 2 >> $rs
    tail "$ot/$ot.txt" -n 1 >> $rs
    echo "" >> $rs
    echo ""
done

