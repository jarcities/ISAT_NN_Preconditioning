python3 trainMLP.py --epochs=500 --bit=64 --num-cpus=0 > out 2> err
cat out | grep "tensor(" | cut -d "(" -f2 | cut -d "," -f1 | cut -d ")" -f1 > out2
