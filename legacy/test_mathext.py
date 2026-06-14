import MathExt
import time

def format_large_number(n_str):
    if len(n_str) <= 20:  # Show full number if it's not too long
        return format(int(n_str), ',')
    else:
        # Show first 10 and last 10 digits with length in middle
        return f"{n_str[:10]}...({len(n_str)} digits)...{n_str[-10:]}"

def test_factorial():
    print("Testing factorial function:")
    
    basic_cases = [0, 1, 5, 10, 20]
    print("\nBasic cases:")
    for n in basic_cases:
        result = MathExt.factorial(n)
        print(f"factorial({n}) = {format_large_number(result)}")
    
    large_cases = [50, 100, 1000]
    print("\nLarge number cases:")
    for n in large_cases:
        start_time = time.time()
        result = MathExt.factorial(n)
        end_time = time.time()
        print(f"factorial({n}) = {format_large_number(result)}")
        print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    
    print("\nTesting error handling:")
    try:
        MathExt.factorial(-1)
    except ValueError as e:
        print("Correctly caught negative input:", str(e))

if __name__ == "__main__":
    test_factorial()