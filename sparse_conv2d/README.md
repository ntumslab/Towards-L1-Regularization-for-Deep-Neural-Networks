# Sparse Convolution Implementation


## Prerequisite

* Python 3.6
* g++


## Install

Install python dependencies:
```bash
pip install -r requirements.txt
```

Compile custom operation:
```bash
make op
```


## Run

Run the dense `VGG-SSH` face detector:
```bash
python prof_dense.py
```

Run the sparse `VGG-SSH-Prune` face detector:
```bash
python prof_sparse.py
```

These scripts will output layer-wise and overall times on your screen.

VGG-SSH:
Timings (microseconds): count=10 first=16030953 curr=15949804 min=15414769 max=16062678 avg=1.58199e+07 std=232476

VGG-SSH-Prune:
Timings (microseconds): count=10 first=4450687 curr=4385284 min=4385284 max=4544607 avg=4.47367e+06 std=46740

The sparse model is 3.5x faster than the dense model on a single core of Intel Xeon Gold 6140 CPU.
