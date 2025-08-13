#/bin/bash

FILES=`ls datasets/*.in`
OUT="out.txt"
NREPS=5

if [[ ! -e "cuda-out" ]] then
    mkdir cuda-out
fi

rm -rf "./cuda-out/*"

for fin in $FILES ;
do
    echo $fin
    otname=`sed 's/datasets\///' <<< "$fin"`
    RES="cuda-out/res-$otname.txt"
    touch $RES
    echo $fin > $RES
    echo -e "t1\tt2\tt3\tt4\tt5" >> $RES
    for rep in `seq $NREPS`; do
        ./cuda-skyline < $fin > $OUT
        TIME=`tail -n 1 $OUT`
        echo -n -e "$TIME\t" >> $RES
    done
    rm $OUT
    echo "" >> $RES
done

