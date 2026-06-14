import re
import matplotlib.pyplot as plt
import numpy as np

def extract_timing_data(filename):
    cpp_times = []
    numpy_times = []
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
            
            cpp_matches = re.findall(r'C\+\+ Implementation Time: (\d+\.\d+) seconds', content)
            numpy_matches = re.findall(r'NumPy Implementation Time: (\d+\.\d+) seconds', content)
            
            cpp_times = np.array([float(time) for time in cpp_matches], dtype=np.float64)
            numpy_times = np.array([float(time) for time in numpy_matches], dtype=np.float64)
            
    except Exception as e:
        print(f"Error while reading file: {str(e)}")
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    return cpp_times, numpy_times

def plot_comparison(cpp_times, numpy_times):
    # Create a new figure with a white background
    plt.figure(figsize=(12, 6), facecolor='white')
    
    # Plot the first 100 measurements for better visibility
    n_points = min(100, len(cpp_times))
    x = np.arange(n_points)
    
    # Plot lines
    plt.plot(x, cpp_times[:n_points], 'b-', label='C++', alpha=0.7)
    plt.plot(x, numpy_times[:n_points], 'g-', label='NumPy', alpha=0.7)
    
    # Add mean lines
    cpp_mean = np.mean(cpp_times)
    numpy_mean = np.mean(numpy_times)
    plt.axhline(y=cpp_mean, color='blue', linestyle='--', alpha=0.5, 
                label=f'C++ Mean: {cpp_mean:.4f}s')
    plt.axhline(y=numpy_mean, color='green', linestyle='--', alpha=0.5, 
                label=f'NumPy Mean: {numpy_mean:.4f}s')
    
    # Customize the plot
    plt.xlabel('Test Number')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance: C++ vs NumPy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show the plot
    try:
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved successfully as 'performance_comparison.png'")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    
    plt.close()

def print_statistics(cpp_times, numpy_times):
    print("\nPerformance Statistics:")
    print("-" * 50)
    print(f"Number of measurements: {len(cpp_times)}")
    print("\nC++ Implementation:")
    print(f"  Mean time: {np.mean(cpp_times):.4f} seconds")
    print(f"  Std dev:  {np.std(cpp_times):.4f} seconds")
    print(f"  Min time: {np.min(cpp_times):.4f} seconds")
    print(f"  Max time: {np.max(cpp_times):.4f} seconds")
    
    print("\nNumPy Implementation:")
    print(f"  Mean time: {np.mean(numpy_times):.4f} seconds")
    print(f"  Std dev:  {np.std(numpy_times):.4f} seconds")
    print(f"  Min time: {np.min(numpy_times):.4f} seconds")
    print(f"  Max time: {np.max(numpy_times):.4f} seconds")
    
    perf_diff = (np.mean(numpy_times) - np.mean(cpp_times)) / np.mean(numpy_times) * 100
    print(f"\nPerformance Difference:")
    print(f"  C++ is {abs(perf_diff):.2f}% {'faster' if perf_diff > 0 else 'slower'} than NumPy")

def main():
    # Extract timing data from the file
    cpp_times, numpy_times = extract_timing_data('matrix_multiplication_results_v2.txt')
    
    if len(cpp_times) > 0 and len(numpy_times) > 0:
        # Create the plot
        plot_comparison(cpp_times, numpy_times)
        
        # Print detailed statistics
        print_statistics(cpp_times, numpy_times)
    else:
        print("No data to analyze")

if __name__ == "__main__":
    main()