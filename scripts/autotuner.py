import re

FILE = "../benchmark_results/kernel_11_autotune_results.txt"
with open(FILE, 'r') as f:
    lines = f.readlines()
    config = ""
    lst, results = [], []
    for line in lines:
        if "NUM_THREADS" in line:
            config = line
            results = []
        if "performance" in line:
            TIME, GFLOPS, SIZE = re.findall(r"\d+\.?\d*", line)
            results.append((TIME, GFLOPS, SIZE))
            if SIZE == "4096":
                lst.append((config, results))
    lst.sort(key=lambda x: float(x[1][5][1]), reverse=True)
    print(lst[0][0])

# kernel 9:(184/576): BK=16 TM=8 TN=4 BM=128 BN=64 NUM_THREADS=256
# kernel 10:(20995/73728): BK=16 BM=64 BN=128 WM=64 WN=32 WN_ITER=1 TM=4 TN=8 NUM_THREADS=128
