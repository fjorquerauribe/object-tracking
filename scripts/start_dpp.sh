#python ../src/test_dpp.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ./data/$1/$2/$3/det/det.txt
#python ../src/test_dpp.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ../data/resnet/$1/$2/$3.txt
python ../src/test_dpp.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ../data/$1-$2/$3.txt