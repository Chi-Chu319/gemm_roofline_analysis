from utils import cdiv

def bytes_to_vgpr(bytes):
    # 32 bits 64 lanes
    return cdiv(cdiv(bytes, 4), 64)

def vgpr_metrics(args, arch_specs, byte_size):
    # a tile, b tile, acc tile
    
    """Calculate workgroup metrics for GEMM operations."""
    M = args.M
    N = args.N
    K = args.K
    num_stages = args.num_stages
    ksplit = args.ksplit
    BLOCK_SIZE_M = args.BLOCK_SIZE_M
    BLOCK_SIZE_N = args.BLOCK_SIZE_N
    BLOCK_SIZE_K = args.BLOCK_SIZE_K

    a_tile_size = BLOCK_SIZE_M * BLOCK_SIZE_K * byte_size
    b_tile_size = BLOCK_SIZE_N * BLOCK_SIZE_K * byte_size
    # fp32 accumulator
    acc_tile_size = BLOCK_SIZE_M * BLOCK_SIZE_N * 4

    print("vgpr Metrics:")
    print(f"a tile vgpr count: {bytes_to_vgpr(a_tile_size)}")
    print(f"b tile vgpr count: {bytes_to_vgpr(b_tile_size)}")
    print(f"acc tile vgpr count: {bytes_to_vgpr(acc_tile_size)}")
    print(f"total vgpr count: {bytes_to_vgpr(a_tile_size + b_tile_size + acc_tile_size)}")
    print(f"max vgpr count: {arch_specs['vgpr']}")
    print(f"max occupancy: {arch_specs['vgpr'] // bytes_to_vgpr(a_tile_size + b_tile_size + acc_tile_size)}")