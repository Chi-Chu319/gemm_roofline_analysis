from utils import cdiv, format_bytes

def lds_metrics(args, arch_specs, byte_size):
    """Calculate workgroup metrics for GEMM operations."""
    M = args.M
    N = args.N
    K = args.K
    num_stages = args.num_stages
    ksplit = args.ksplit
    BLOCK_SIZE_M = args.BLOCK_SIZE_M
    BLOCK_SIZE_N = args.BLOCK_SIZE_N
    BLOCK_SIZE_K = args.BLOCK_SIZE_K

    tile_size_a = BLOCK_SIZE_M * BLOCK_SIZE_K * byte_size
    tile_size_b = BLOCK_SIZE_N * BLOCK_SIZE_K * byte_size
    tile_size = tile_size_a + tile_size_b

    lds_size = arch_specs["LDS"]

    print("LDS Metrics:")
    print(f"LDS size: {format_bytes(lds_size)}")
    print(f"tile size: {format_bytes(tile_size)}")
    print(f"num stages: {num_stages}")
    print(f"pipelined tile size: {format_bytes(tile_size * num_stages)}")
    print(f"max tiles per LDS: {lds_size // tile_size}")
