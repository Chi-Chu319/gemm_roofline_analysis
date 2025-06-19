import argparse
import json
import os
from metrics.l2 import l2_metrics
from metrics.lds import lds_metrics
from metrics.vgpr import vgpr_metrics
from metrics.workgroup import workgroup_metrics

dtypes_size = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "f64": 8,
    "int8": 1,
    "int4": 0.5,
    "fp4": 0.5,
    "int16": 2,
    "int32": 4,
}

metrices = [
    workgroup_metrics,
    l2_metrics,
    lds_metrics,
    vgpr_metrics,
]

def roofline_analysis(args, arch_specs, byte_size):
    print("Roofline Analysis:")
    print(f"Matrix dimensions: M={args.M}, N={args.N}, K={args.K}")
    print(f"Block sizes: BLOCK_SIZE_M={args.BLOCK_SIZE_M}, BLOCK_SIZE_N={args.BLOCK_SIZE_N}, BLOCK_SIZE_K={args.BLOCK_SIZE_K}") 
    print(f"Number of stages: {args.num_stages}, ksplit: {args.ksplit}, dtype: {args.dtype}, arch: {args.arch}")
    print(f"Byte size for dtype {args.dtype}: {byte_size} bytes")
    print(f"Architecture specs: {args.arch}")

    for metric in metrices:
        print(f"--------------------------")
        metric(args, arch_specs, byte_size)

    print(f"--------------------------")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Roofline analysis for GEMM operations"
    )
    
    # Add arguments for matrix dimensions
    parser.add_argument("-M", "--M", type=int, required=True,
                        help="Number of rows in matrix A")
    parser.add_argument("-N", "--N", type=int, required=True,
                        help="Number of columns in matrix B")
    parser.add_argument("-K", "--K", type=int, required=True,
                        help="Number of columns in matrix A / rows in matrix B")
    
    # Add arguments for blocking parameters
    parser.add_argument("--BLOCK_SIZE_M", type=int, required=True,
                        help="Block size for M dimension")
    parser.add_argument("--BLOCK_SIZE_N", type=int, required=True,
                        help="Block size for N dimension")
    parser.add_argument("--BLOCK_SIZE_K", type=int, required=True,
                        help="Block size for K dimension")
    parser.add_argument("--num_stages", type=int, required=False, default=1,
                        help="num_stages")
    parser.add_argument("--ksplit", type=int, required=False, default=1,
                        help="ksplit")
    parser.add_argument("--dtype", type=str, required=False, default="fp32",
                        help="dtype")
    parser.add_argument("--arch", type=str, required=False, default="gfx950",
                    help="arch")
    
    return parser.parse_args()

def args_preprocess(args):
    """Preprocess arguments for roofline analysis."""
    arch = args.arch

    current_dir = os.path.dirname(os.path.abspath(__file__))
    specs_path = os.path.join(current_dir, 'specs', f'{arch}.json')

    # Read and parse the JSON file
    try:
        with open(specs_path, 'r') as file:
            arch_specs = json.load(file)
    except FileNotFoundError:
        print(f"Error: Architecture specs file not found at {specs_path}")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON data from {specs_path}")
    
    if args.dtype not in arch_specs['dtypes']:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Supported dtypes are: {list(arch_specs['dtypes'].keys())}")

    return args, arch_specs

if __name__ == "__main__":
    args, arch_specs = args_preprocess(parse_args())
    
    byte_size = dtypes_size[args.dtype]
    roofline_analysis(parse_args(), arch_specs, byte_size)