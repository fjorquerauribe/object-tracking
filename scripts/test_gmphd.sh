declare -a datasets=(MOT16)
declare -a types=(train)
option=$1
OUTPUT_DIRECTORY=$2

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        mkdir -p ./results/$OUTPUT_DIRECTORY/$dataset/$type/
        ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
        sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
        while read sequence;
        do
            echo $dataset,$sequence
            /bin/bash $PWD/start_gmphd.sh $dataset $type $sequence $option > ./results/$OUTPUT_DIRECTORY/$dataset/$type/$sequence.txt
        done <./data/$dataset/$type/sequences.lst
    done
done
