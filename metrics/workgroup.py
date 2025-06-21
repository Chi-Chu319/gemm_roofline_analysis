from utils import cdiv

def workgroup_metrics(args, arch_specs, byte_size):
    """Calculate workgroup metrics for GEMM operations."""
    M = args.M
    N = args.N
    K = args.K
    num_stages = args.num_stages
    ksplit = args.ksplit
    BLOCK_SIZE_M = args.BLOCK_SIZE_M
    BLOCK_SIZE_N = args.BLOCK_SIZE_N
    BLOCK_SIZE_K = args.BLOCK_SIZE_K

    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    num_iter_k = cdiv(cdiv(K, ksplit), BLOCK_SIZE_K)
    num_iter_stages = cdiv(cdiv(K, ksplit), num_stages * BLOCK_SIZE_K)
    num_cus = arch_specs["num_cus"]

    print("Workgroup Metrics:")
    print(f"num_pid_m: {num_pid_m}")
    print(f"num_pid_n: {num_pid_n}")
    print(f"num waves: {num_pid_m * num_pid_n * ksplit}")
    print(f"num max waves per CU: {cdiv((num_pid_m * num_pid_n * ksplit) , num_cus)} of {num_cus} CUs")
    print(f"number of k iteration: {num_iter_k}")
    print(f"number of k iteration with {num_stages}-stage unrolling: {num_iter_stages}")
