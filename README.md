# gemm_roofline_analysis
Rocm gemm roofline analysis. It gives estimates of:
num of workgroups, registers, LDS, iterations in the workgroups, extimated memory load


usage:
```
python3 gemm_roofline_analysis.py -M 16384 -N 16284 -K 2048 --BLOCK_SIZE_M 256 --BLOCK_SIZE_N 256 --BLOCK_SIZE_K 256 --dtype fp4
```

example output:
```
Roofline Analysis:
Matrix dimensions: M=16384, N=16384, K=16384
Block sizes: BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256
Number of stages: 1, ksplit: 1, dtype: fp4, arch: gfx950
Byte size for dtype fp4: 0.5 bytes
Architecture specs: gfx950
--------------------------
Workgroup Metrics:
num_pid_m: 64
num_pid_n: 64
num waves: 4096
num max waves per CU: 16 of 256 CUs
number of k iteration: 64
number of k iteration with 1-stage unrolling: 64
--------------------------
l2 Metrics:
l2 size: 4.00 MB
pid load size (for A and B): 4.00 MB
max pid per l2: 1.0
--------------------------
LDS Metrics:
LDS size: 160.00 KB
tile size: 64.00 KB
num stages: 1
pipelined tile size: 64.00 KB
max tiles per LDS: 2.0
--------------------------
vgpr Metrics:
a tile vgpr count: 128.0
b tile vgpr count: 128.0
acc tile vgpr count: 1024
total vgpr count: 256.0 (we assume the acc vgpr is written immediately to the vram no not counted here)
max vgpr count: 512
max occupancy: 2.0
--------------------------
```
