
#!/bin/bash

# Usage: ./lu.sh <N> <threads> <method>
# Example usage:
# ./lu.sh 200 8 serial
# ./lu.sh 200 8 naive
# ./lu.sh 200 8 optimal

# 1. Check accuracy for different methods (serial, naive, optimal)
echo "Step 1: Checking accuracy"

make clean && make
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"


echo "Running Valgrind to check for memory leaks..."
valgrind --leak-check=full --error-exitcode=1 ./lu 200 8 serial > valgrind_output.txt 2>&1
valgrind --leak-check=full --error-exitcode=1 ./lu 200 8 naive > valgrind_output.txt 2>&1
valgrind --leak-check=full --error-exitcode=1 ./lu 200 8 optimal > valgrind_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "No memory leaks detected."
else
    echo "Memory leaks detected! Check valgrind_output.txt"
fi

# Accuracy check
for N in 2 3 10 100 1000; do
    for method in serial naive optimal; do
        echo "Checking accuracy for method $method with N = $N..."
        ./lu $N 1 $method > result_$method_$N.txt
        
        # Check for correctness message
        if grep -q "LU decomposition is correct" result_$method_$N.txt; then
            echo "Method $method with N = $N: Correct"
        else
            echo "Method $method with N = $N: Incorrect"
        fi
    done
done


#stop here to disable check part
# 2. Measure execution time for different methods and input sizes
echo "Step 2: Measuring execution time for each method"
> time.txt  
for N in 10 50 100 200 500 1000 2000 3000 4000 6000 ; do
    for method in naive optimal; do
        echo "Measuring time for method $method with N = $N and threads = 8 ..."
        output=$(./lu $N 8 $method)

       
        time_taken=$(echo "$output" | grep "Time taken for LU decomposition:" | awk '{print $6}')
        
       
        if [ -n "$time_taken" ]; then
            echo "($N, $method, $time_taken)" >> time.txt
        else
            echo "Warning: Time not found for N=$N, method=$method"
        fi
    done
done

# 5. Check data locality (using gprof or other profiling tools)
echo "Step 5: Checking data locality"
    for method in naive optimal; do
        echo "Checking data locality for method $method with threads = $threads..."
        gprof ./lu 1000 $threads $method > gprof_data_locality_$method.txt
        # Alternatively, use other tools like `perf` for better data locality profiling
        # perf stat -e cache-references,cache-misses ./lu $N $threads $method
    done

# 6. Load balance check
echo "Step 6: Checking load balance"
# You can analyze load balance using the gprof data, or you can manually check the output of gprof for uneven distribution.



