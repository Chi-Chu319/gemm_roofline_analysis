# gemm_roofline_analysis
Rocm gemm roofline analysis. It gives estimates of:
num of workgroups, registers, LDS, iterations in the workgroups, extimated memory load


usage:
```
python3 gemm_roofline_analysis.py -M 16384 -N 16284 -K 2048 --BLOCK_SIZE_M 256 --BLOCK_SIZE_N 256 --BLOCK_SIZE_K 256 --dtype fp4
```