import gc
import sys
import time
import argparse
import platform
import psutil
import MathExt
import numpy as np
from typing import List, Tuple, Optional


def get_system_info():
    """Collect detailed system information"""
    cpu_info = platform.processor()
    try:
        cpu_freq = psutil.cpu_freq().current
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
    except:
        cpu_freq = "Unknown"
        cpu_count = "Unknown"
        cpu_logical = "Unknown"
    
    memory = psutil.virtual_memory()
    
    return {
        "cpu_info": cpu_info,
        "cpu_freq": f"{cpu_freq} MHz" if isinstance(cpu_freq, (int, float)) else cpu_freq,
        "cpu_cores": f"{cpu_count} physical, {cpu_logical} logical" if isinstance(cpu_count, int) else cpu_count,
        "memory_total": f"{memory.total / (1024**3):.2f} GB",
        "memory_available": f"{memory.available / (1024**3):.2f} GB",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blas_info": "Available" if hasattr(np, '__config__') else "Unknown"
    }


def warm_up_cpu():
    """Warm up the CPU to ensure consistent performance"""
    print("Warming up CPU...", end="", flush=True)
    
    A = np.random.rand(500, 500).tolist()
    B = np.random.rand(500, 500).tolist()
    
    for _ in range(5):
        _ = MathExt.matrix_multiply(A, B)
        _ = np.dot(A, B)
        gc.collect()
    
    print(" Done")


def run_single_test(size: int, warmup: bool = True, tolerance: float = 1e-10) -> Tuple[float, float, bool]:
    """Run a single matrix multiplication test of specified size"""
    if warmup:
        _ = MathExt.matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        _ = np.dot(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]))
    
    print(f"Generating {size}x{size} matrices...", end="", flush=True)
    A = np.random.rand(size, size).tolist()
    B = np.random.rand(size, size).tolist()
    print(" Done")
    
    gc.collect()
    
    print(f"Running C++ implementation...", end="", flush=True)
    start_time = time.time()
    result = MathExt.matrix_multiply(A, B)
    cpp_time = time.time() - start_time
    print(f" Done in {cpp_time:.4f}s")
    
    gc.collect()
    
    print(f"Running NumPy implementation...", end="", flush=True)
    start_time = time.time()
    np_result = np.dot(A, B)
    numpy_time = time.time() - start_time
    print(f" Done in {numpy_time:.4f}s")
    
    print("Verifying results...", end="", flush=True)
    results_match = np.allclose(result, np_result, rtol=tolerance, atol=tolerance)
    print(" Match" if results_match else " MISMATCH!")
    
    return cpp_time, numpy_time, results_match


def test_matrix_multiplication(sizes: Optional[List[int]] = None, 
                              non_square: bool = True, 
                              small_test: bool = True,
                              tolerance: float = 1e-10):
    try:
        print("\n" + "="*70)
        print(f"MATRIX MULTIPLICATION TEST SUITE")
        print("="*70)
        
        info = get_system_info()
        print("\nSystem Information:")
        print(f"Day and Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.architecture()[0]}")
        print(f"Processor: {platform.processor()}")
        print(f"CPU: {info['cpu_info']}")
        print(f"CPU Frequency: {info['cpu_freq']}")
        print(f"CPU Cores: {info['cpu_cores']}")
        print(f"Memory: {info['memory_total']} (Available: {info['memory_available']})")
        print(f"Python: {info['python_version']}")
        print(f"NumPy: {info['numpy_version']}")
        
        warm_up_cpu()
        
        if small_test:
            print("\nTesting small matrix multiplication:")
            A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            B = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
            
            result = MathExt.matrix_multiply(A, B)
            np_result = np.dot(A, B)
            
            print("C++ Result:", result)
            print("NumPy Result:", np_result.tolist())
            print("Match:", np.allclose(result, np_result, rtol=tolerance, atol=tolerance))
        
        if sizes is None:
            sizes = [100, 250, 500, 1000, 1500, 1750]
        
        print("\nLarge matrix performance tests:")
        results = {}
        
        for size in sizes:
            print(f"\n{'-'*60}")
            print(f"TESTING {size}x{size} MATRICES")
            print(f"{'-'*60}")
            
            try:
                available_mem = psutil.virtual_memory().available
                required_mem = size * size * 8 * 4  # Rough estimate: two input matrices + two output matrices, 8 bytes per double
                
                if required_mem > available_mem * 0.9:  # Use 90% of available memory as threshold
                    print(f"WARNING: Matrix size {size}x{size} may exceed available memory.")
                    print(f"Required: ~{required_mem/(1024**3):.2f} GB, Available: {available_mem/(1024**3):.2f} GB")
                    user_continue = input("Continue anyway? (y/n): ")
                    if user_continue.lower() != 'y':
                        print("Skipping this test.")
                        continue
                
                cpp_time, numpy_time, results_match = run_single_test(size, warmup=True, tolerance=tolerance)
                
                print(f"\nC++ Implementation Time: {cpp_time:.4f} seconds")
                print(f"NumPy Implementation Time: {numpy_time:.4f} seconds")
                print(f"Results Match: {results_match}")
                
                if cpp_time > 0:
                    speed_ratio = numpy_time / cpp_time
                    print(f"Speed Ratio (NumPy/C++): {speed_ratio:.2f}")
                    print(f"C++ is {(speed_ratio - 1) * 100:.2f}% {'faster' if speed_ratio > 1 else 'slower'} than NumPy")
                
                results[size] = {
                    'cpp_time': cpp_time,
                    'numpy_time': numpy_time,
                    'match': results_match,
                    'ratio': numpy_time / cpp_time if cpp_time > 0 else 0
                }
                
            except Exception as e:
                print(f"Error testing {size}x{size} matrices: {str(e)}")
            
            gc.collect()
        
        if non_square:
            print("\nNon-square matrix test:")
            A = np.random.rand(100, 200).tolist()
            B = np.random.rand(200, 50).tolist()
            
            start_time = time.time()
            result = MathExt.matrix_multiply(A, B)
            cpp_time = time.time() - start_time
            
            start_time = time.time()
            np_result = np.dot(A, B)
            numpy_time = time.time() - start_time
            
            print(f"C++ Implementation Time: {cpp_time:.4f} seconds")
            print(f"NumPy Implementation Time: {numpy_time:.4f} seconds")
            print(f"Results Match: {np.allclose(result, np_result, rtol=tolerance, atol=tolerance)}")
            
            if cpp_time > 0:
                print(f"Speed Ratio (NumPy/C++): {numpy_time/cpp_time:.2f}")
        
        return results

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test matrix multiplication performance.")
    parser.add_argument("sizes", type=int, nargs="*", 
                        help="Matrix sizes to test (e.g., 1000 1500 1750)")
    parser.add_argument("--no-small", action="store_true", help="Skip small matrix test")
    parser.add_argument("--no-nonsquare", action="store_true", help="Skip non-square matrix test")
    parser.add_argument("--tolerance", type=float, default=1e-10, 
                        help="Tolerance for result verification (default: 1e-10)")
    args = parser.parse_args()
    
    test_sizes = args.sizes if args.sizes else None
    test_matrix_multiplication(
        sizes=test_sizes,
        non_square=not args.no_nonsquare,
        small_test=not args.no_small,
        tolerance=args.tolerance
    )