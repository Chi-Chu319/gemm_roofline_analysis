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
Matrix dimensions: M=16384, N=16284, K=2048
Block sizes: BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256
Number of stages: 1, ksplit: 1, dtype: fp4, arch: gfx950
Byte size for dtype fp4: 0.5 bytes
Architecture specs: gfx950
--------------------------
Workgroup Metrics:
num_pid_m: 64
num_pid_n: 64
num waves: 4096
num waves per CU: 8
num_iter_k: 8
num_iter_stages: 8
--------------------------
l2 Metrics:
l2 size: 4.00 MBB
pid load size (for A and B): 512.00 KBB
max pid per l2: 8.0
--------------------------
LDS Metrics:
LDS size: 160.00 KBB
tile size: 64.00 KBB
num stages: 1
pipelined tile size: 64.00 KBB
max tiles per LDS: 2.0
--------------------------
vgpr Metrics:
a tile vgpr count: 128.0
b tile vgpr count: 128.0
acc tile vgpr count: 1024
total vgpr count: 1280.0
max vgpr count: 512
max occupancy: 0.0
--------------------------
```