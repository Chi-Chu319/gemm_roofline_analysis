from utils import cdiv, format_bytes
def l2_metrics(args, arch_specs, byte_size):
    """Calculate workgroup metrics for GEMM operations."""
    K = args.K
    ksplit = args.ksplit
    BLOCK_SIZE_M = args.BLOCK_SIZE_M
    BLOCK_SIZE_N = args.BLOCK_SIZE_N

    pid_size_a = (BLOCK_SIZE_M * cdiv(K, ksplit) * byte_size)
    pid_size_b = (BLOCK_SIZE_N * cdiv(K, ksplit) * byte_size)
    pid_size = pid_size_a + pid_size_b
    l2_size = arch_specs["L2"]

    print("l2 Metrics:")
    print(f"l2 size: {format_bytes(l2_size)}")
    print(f"pid load size (for A and B): {format_bytes(pid_size)}")
    print(f"max pid per l2: {l2_size // pid_size}")