python3 trainMLP_cpu.py --epochs=100 --bit=64 --num-cpus=4 --use_cpu > out 2> err
cat out | grep "tensor(" | cut -d "(" -f2 | cut -d "," -f1 | cut -d ")" -f1 > out2
