import numpy as np
import MathExt
import time
import gc
import os


def save_results_to_file(results, filename="matrix_multiplication_results_v2.txt"):
    # if the file does not exist, create it and write the results
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write(results + "\n")

    with open(filename, "a") as file:
        file.write(results + "\n")


def test_matrix_multiplication():
    # Test case 1: Small square matrices
    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    # result = MathExt.matrix_multiply(A, B)
    # np_result = np.dot(A, B)
    
    # results = (
    #     "Small matrix test:\n"
    #     f"C++ Result: {result}\n"
    #     f"NumPy Result: {np_result.tolist()}\n"
    #     f"Match: {np.allclose(result, np_result)}\n"
    # )
    # print(results)
    # save_results_to_file(results)
    
    sizes = [1000]
    for size in sizes:
        A = np.random.rand(size, size).tolist()
        B = np.random.rand(size, size).tolist()
        
        # Time C++ implementation
        start_time = time.time()
        cpp_result = MathExt.matrix_multiply(A, B)
        cpp_time = time.time() - start_time
        
        # Time NumPy implementation
        start_time = time.time()
        np_result = np.dot(A, B)
        np_time = time.time() - start_time
        
        results = (
            f"\nLarge matrix ({size}x{size}) performance test:\n"
            f"C++ Implementation Time: {cpp_time:.4f} seconds\n"
            f"NumPy Implementation Time: {np_time:.4f} seconds\n"
            f"Results Match: {np.allclose(cpp_result, np_result)}\n"
            f"Speed Ratio (NumPy/C++): {np_time/cpp_time:.2f}\n"
        )
        print(results)
        save_results_to_file(results)
        gc.collect()
    
    # Test case 3: Non-square matrices
    # A = np.random.rand(100, 200).tolist()
    # B = np.random.rand(200, 50).tolist()
    
    # result = MathExt.matrix_multiply(A, B)
    # np_result = np.dot(A, B)
    
    # results = (
    #     "\nNon-square matrix test:\n"
    #     f"Results Match: {np.allclose(result, np_result)}\n"
    # )
    # print(results)
    # save_results_to_file(results)



def warmup():
    size = 500
    A = np.random.rand(size, size).tolist()
    B = np.random.rand(size, size).tolist()
    _ = MathExt.matrix_multiply(A, B)
    _ = np.dot(np.array(A), np.array(B))


if __name__ == "__main__":
    warmup()
    test_matrix_multiplication()

# python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py && python test.py
