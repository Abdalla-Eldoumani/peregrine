import subprocess
import sys
import os
import numpy as np
import re
import time
import argparse
import gc
import json
import platform
import datetime
from typing import List, Dict, Any, Optional


def clear_memory():
    """Force garbage collection and clear memory"""
    gc.collect()
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except:
            pass


def set_process_priority(high_priority=True):
    """Set process priority for more consistent benchmarking"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        
        if sys.platform == 'win32':
            if high_priority:
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                process.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:
            # For Unix systems
            if high_priority:
                process.nice(-10)
            else:
                process.nice(0)
        return True
    except:
        return False


def detect_outliers(data, threshold=1.5):
    """Detect outliers using IQR method"""
    if len(data) < 4:
        return [], data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return outliers, filtered_data


def print_histogram(data, bins=10, width=50):
    """Print a simple ASCII histogram"""
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    
    print("\nDistribution:")
    for i in range(len(hist)):
        bar_len = int(width * hist[i] / max_count) if max_count > 0 else 0
        print(f"{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}: {'#' * bar_len} ({hist[i]})")


def run_benchmark(iterations: int, sizes: List[int], 
                 detailed: bool = False,
                 warmup: bool = True,
                 save_results: Optional[str] = None,
                 show_histograms: bool = False,
                 outlier_detection: bool = True):
    """Run comprehensive matrix multiplication benchmarks"""
    results = {}
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'command': ' '.join(sys.argv),
    }
    
    priority_set = set_process_priority(True)
    if priority_set:
        print("Process priority increased for more consistent benchmarking")
    
    if warmup:
        print("Running initial warmup...")
        subprocess.run([sys.executable, "test_matrix.py", "100"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        clear_memory()
        print("Warmup complete")
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {size}x{size} MATRICES")
        print(f"{'='*60}")
        
        cpp_times = []
        numpy_times = []
        
        print(f"Running benchmark for {iterations} iterations...")
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r")
            
            cmd = [sys.executable, "test_matrix.py", str(size), "--no-small", "--no-nonsquare"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            output = result.stdout
            
            if "Error occurred:" in output:
                error_match = re.search(r"Error occurred: (.*?)$", output, re.MULTILINE)
                error_msg = error_match.group(1) if error_match else "Unknown error"
                print(f"\nError in iteration {i+1}: {error_msg}")
                continue
            
            match_cpp = re.search(f"C\\+\\+ Implementation Time: ([\d\.]+) seconds", output)
            match_numpy = re.search(f"NumPy Implementation Time: ([\d\.]+) seconds", output)
            
            if match_cpp and match_numpy:
                cpp_time = float(match_cpp.group(1))
                numpy_time = float(match_numpy.group(1))
                
                cpp_times.append(cpp_time)
                numpy_times.append(numpy_time)
            else:
                print(f"\nWarning: Could not extract timing info from iteration {i+1}")
            
            clear_memory()
        
        if len(cpp_times) < iterations * 0.5:
            print(f"\nWarning: Only {len(cpp_times)} out of {iterations} iterations produced usable results")
        
        if cpp_times and numpy_times:
            if outlier_detection and len(cpp_times) >= 5:
                cpp_outliers, cpp_filtered = detect_outliers(cpp_times)
                numpy_outliers, numpy_filtered = detect_outliers(numpy_times)
                
                if cpp_outliers:
                    print(f"\nDetected {len(cpp_outliers)} outliers in C++ times, removing them")
                    if detailed:
                        print(f"Outliers: {cpp_outliers}")
                
                if numpy_outliers:
                    print(f"Detected {len(numpy_outliers)} outliers in NumPy times, removing them")
                    if detailed:
                        print(f"Outliers: {numpy_outliers}")
                
                cpp_times = cpp_filtered if cpp_filtered else cpp_times
                numpy_times = numpy_filtered if numpy_filtered else numpy_times
            
            # Calculate statistics
            cpp_stats = {
                "mean": np.mean(cpp_times),
                "median": np.median(cpp_times),
                "std": np.std(cpp_times),
                "min": np.min(cpp_times),
                "max": np.max(cpp_times),
                "p25": np.percentile(cpp_times, 25),
                "p75": np.percentile(cpp_times, 75),
                "samples": len(cpp_times)
            }
            
            numpy_stats = {
                "mean": np.mean(numpy_times),
                "median": np.median(numpy_times),
                "std": np.std(numpy_times),
                "min": np.min(numpy_times),
                "max": np.max(numpy_times),
                "p25": np.percentile(numpy_times, 25),
                "p75": np.percentile(numpy_times, 75),
                "samples": len(numpy_times)
            }
            
            if show_histograms:
                print("\nC++ time distribution:")
                print_histogram(cpp_times)
                print("\nNumPy time distribution:")
                print_histogram(numpy_times)
            
            if detailed:
                cpp_stats["all_times"] = cpp_times
                numpy_stats["all_times"] = numpy_times
            
            results[size] = {
                "cpp": cpp_stats,
                "numpy": numpy_stats,
            }
    
    if priority_set:
        set_process_priority(False)
        print("\nProcess priority restored to normal")
    
    print_benchmark_results(results)
    
    if save_results:
        full_results = {
            "system_info": system_info,
            "results": results
        }
        try:
            filename = save_results if save_results.endswith('.json') else f"{save_results}.json"
            with open(filename, 'w') as f:
                json.dump(full_results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"\nError saving results: {str(e)}")
    
    return results


def print_benchmark_results(results):
    """Print formatted benchmark results"""
    print("\n" + "="*80)
    print(f"BENCHMARK RESULTS")
    print("="*80)
    
    for size, data in results.items():
        print(f"\n### Performance Statistics ({size}x{size} matrices)\n")
        
        print("| Implementation | Mean Time  | Median Time | Std Dev   | Min Time  | Max Time  | Samples |")
        print("| -------------- | ---------- | ----------- | --------- | --------- | --------- | ------- |")
        print(f"| **C++**        | {data['cpp']['mean']:.4f} s   | {data['cpp']['median']:.4f} s    | {data['cpp']['std']:.4f} s  | {data['cpp']['min']:.4f} s  | {data['cpp']['max']:.4f} s  | {data['cpp']['samples']} |")
        print(f"| **NumPy**      | {data['numpy']['mean']:.4f} s   | {data['numpy']['median']:.4f} s    | {data['numpy']['std']:.4f} s  | {data['numpy']['min']:.4f} s  | {data['numpy']['max']:.4f} s  | {data['numpy']['samples']} |")
        
        improvement_mean = (data['numpy']['mean'] - data['cpp']['mean']) / data['numpy']['mean'] * 100
        improvement_median = (data['numpy']['median'] - data['cpp']['median']) / data['numpy']['median'] * 100
        
        print(f"\n- C++ is ~{improvement_mean:.2f}% faster on average (mean)")
        print(f"- C++ is ~{improvement_median:.2f}% faster at median execution time")
        print(f"- {'More' if data['cpp']['std'] < data['numpy']['std'] else 'Less'} consistent performance (lower standard deviation)")
        print(f"- {'Faster' if data['cpp']['max'] < data['numpy']['max'] else 'Slower'} worst-case performance (lower max time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive matrix multiplication benchmarks.")
    parser.add_argument("iterations", type=int, nargs="?", default=10,
                        help="Number of iterations to run per matrix size (default: 10)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 1500, 1750],
                        help="Matrix sizes to benchmark (default: 1000 1500 1750)")
    parser.add_argument("--detailed", action="store_true",
                        help="Include all raw timing data in output")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip initial warmup run")
    parser.add_argument("--save", type=str, metavar="FILENAME",
                        help="Save results to a JSON file")
    parser.add_argument("--histograms", action="store_true",
                        help="Show time distribution histograms")
    parser.add_argument("--no-outlier-detection", action="store_true",
                        help="Disable outlier detection and filtering")
    
    args = parser.parse_args()
    
    run_benchmark(
        iterations=args.iterations,
        sizes=args.sizes,
        detailed=args.detailed,
        warmup=not args.no_warmup,
        save_results=args.save,
        show_histograms=args.histograms,
        outlier_detection=not args.no_outlier_detection
    )