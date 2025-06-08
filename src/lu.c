/*
 * The project is a collaborative effort of me as well as the usage
 * of the Internet and AI, including ChatGPT and Deepseek AI.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define SERIAL_THRESHOLD 100;  // serial threshold


/**
 * Initializes matrices A and A_decompos with random values ensuring diagonal dominance
 * @para A: Pointer to original matrix (output parameter)
 * @para A_decompos_row: Pointer to decomposition matrix (output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void initialize_row(double **A, double **A_decompos_row, int N)
{
    srand(time(NULL)); 

    // 
    for (int i = 0; i < N; i++)
    {
        A[i] = (double *)malloc(N * sizeof(double));
        A_decompos_row[i] = (double *)malloc(N * sizeof(double));
    }

    // random value
    for (int i = 0; i < N; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            if (i != j)
            {
                A[i][j] = (double)rand() / (double)RAND_MAX; 
                row_sum += fabs(A[i][j]);
            }
        }
        /// Ensure diagonal dominance
        A[i][i] = row_sum + 1.0; // |A[i][i]| > ∑|A[i][j]|
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // A[i][j]->A_row[i*N+j]
            A_decompos_row[i][j] = A[i][j];
        }
    }
}

/**
 * Initializes matrices A and A_decompos with random values ensuring diagonal dominance
 * @para A: Pointer to original matrix (output parameter)
 * @para A_decompos: Pointer to decomposition matrix (output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void initialize_column(double **A, double *A_decompos, int N)
{
    srand(time(NULL)); 

    // 
    for (int i = 0; i < N; i++)
    {
        A[i] = (double *)malloc(N * sizeof(double));
    }

    // random value
    for (int i = 0; i < N; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            if (i != j)
            {
                A[i][j] = (double)rand() / (double)RAND_MAX; 
                row_sum += fabs(A[i][j]);
            }
        }
        /// Ensure diagonal dominance
        A[i][i] = row_sum + 1.0; // |A[i][i]| > ∑|A[i][j]|
    }

    // column-major order
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            // A[i][j]->A_col[i + j*N]
            A_decompos[i + j * N] = A[i][j];
        }
    }
}
/**
 * Verifies LU decomposition correctness by matrix multiplication check
 * @para A: Original matrix (input parameter)
 * @para A_decompos: Decomposed matrix (input parameter)
 * @para N: Matrix dimension (input parameter)
 * @return: 1 if correct, 0 otherwise
 * @brief with help from ChatGPT
 */
int check_correctness(double **A, double **A_decompos, int N)
{
    // Verify correctness by checking if L * U = A
    double **L = (double **)malloc(N * sizeof(double *));
    double **U = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        L[i] = (double *)malloc(N * sizeof(double));
        U[i] = (double *)malloc(N * sizeof(double));
    }
    // Extract L and U matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i > j)
            {
                L[i][j] = A_decompos[i][j];
                U[i][j] = 0.0;
            }
            else if (i == j)
            {
                L[i][j] = 1.0;
                U[i][j] = A_decompos[i][j];
            }
            else
            {
                L[i][j] = 0.0;
                U[i][j] = A_decompos[i][j];
            }
        }
    }
    // Multiply L and U and check if it equals the original matrix A
    double **LU = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        LU[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++)
        {
            LU[i][j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    // Check if LU is equal to A (within a small tolerance)
    int is_correct = 1;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(LU[i][j] - A[i][j]) > 1e-9)
            {
                is_correct = 0;
                break;
            }
        }
        if (!is_correct)
            break;
    }
    // Free allocated memory
    for (int i = 0; i < N; i++)
    {
        free(L[i]);
        free(U[i]);
        free(LU[i]);
    }
    free(L);
    free(U);
    free(LU);
    return is_correct;
}

int check_correctness_col(double **A, double *A_decompos, int N)
{
    // Verify correctness by checking if L * U = A
    double **L = (double **)malloc(N * sizeof(double *));
    double **U = (double **)malloc(N * sizeof(double *));
    double **A_detransform = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        L[i] = (double *)malloc(N * sizeof(double));
        U[i] = (double *)malloc(N * sizeof(double));
        A_detransform[i] = (double *)malloc(N * sizeof(double));

    }

    //detranstion to row-major order
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            A_detransform[i][j] = A_decompos[i + j * N];
        }
    }

    // Extract L and U matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i > j)
            {
                L[i][j] = A_detransform[i][j];
                U[i][j] = 0.0;
            }
            else if (i == j)
            {
                L[i][j] = 1.0;
                U[i][j] = A_detransform[i][j];
            }
            else
            {
                L[i][j] = 0.0;
                U[i][j] = A_detransform[i][j];
            }
        }
    }

    // Multiply L and U and check if it equals the original matrix A
    double **LU = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        LU[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++)
        {
            LU[i][j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }

    // Check if LU is equal to A (within a small tolerance)
    int is_correct = 1;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(LU[i][j] - A[i][j]) > 1e-9)
            {
                is_correct = 0;
                break;
            }
        }
        if (!is_correct)
            break;
    }

    // Free allocated memory
    for (int i = 0; i < N; i++)
    {
        free(L[i]);
        free(U[i]);
        free(LU[i]);
        free(A_detransform[i]);
    }
    free(L);
    free(U);
    free(LU);
    free(A_detransform);

    return is_correct;
}

/**
 * Performs sequential LU decomposition using Doolittle's algorithm
 * @para A_decompos: Matrix to decompose (input/output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void LU_direct(double **A_decompos, int N)
{

    for (int k = 0; k < N; k++)
    {
        if (A_decompos[k][k] == 0.0)
        {
            printf("Zero pivot detected at row %d! LU decomposition failed.\n", k);
            exit(1);
        }

        for (int i = k + 1; i < N; i++)
        {
            A_decompos[i][k] /= A_decompos[k][k];
        }


        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A_decompos[i][j] -= A_decompos[i][k] * A_decompos[k][j];
            }
        }
    }
}

/**
 * Performs naive LU decomposition using Doolittle's algorithm
 * @para A_decompos: Matrix to decompose (input/output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void LU_naive(double **A_decompos, int N)
{
    for (int k = 0; k < N; k++)
    {
        if (A_decompos[k][k] == 0.0)
        {
            printf("Zero pivot detected at row %d! LU decomposition failed.\n", k);
            exit(1);
        }
        
        for (int i = k + 1; i < N; i++)
        {
            A_decompos[i][k] /= A_decompos[k][k];
        }

// paralleling part
#pragma omp parallel for
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A_decompos[i][j] -= A_decompos[i][k] * A_decompos[k][j];
            }
        }
    }
}

/**
 * Performs LU decomposition using Doolittle's algorithm with column-major order
 * @para A_decompos: Matrix to decompose (input/output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void LU_direct_col(double *A_decompos, int N, int start)
{

    if (start >= N)
    {
        return; // No more columns to process
    }
    else if (start <= 0)
    {

        //since the first column is already done, we need to finish the 1 rest

        for (int i = 1; i < N; i++)
        {
            for (int j = 1; j < N; j++)
            {
                A_decompos[i + j * N] -= A_decompos[i] * A_decompos[j * N];
            }
        }
        start = 1; // Start from the `second` column
    }

    //do LU decomposition from k col
    for (int k = start; k < N; k++)
    {

        for (int i = k + 1; i < N; i++)
        {
            double pivot = 1.0/A_decompos[k + k * N];
            A_decompos[i + k * N] *= pivot;
        }


        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A_decompos[i + j * N] -= A_decompos[i + k * N] * A_decompos[k + j * N];
            }
        }
    }
}


/**
 * Parallel LU decomposition with well OpenMP implementation
 * serial swith
 * @para A_decompos: Matrix to decompose (input/output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void LU_opt(double *A_decompos, int N)
{
    omp_lock_t *locks = (omp_lock_t *)malloc(N * sizeof(omp_lock_t));
    for (int i = 0; i < N; ++i)
    {
        omp_init_lock(&locks[i]);
    }
    int nlim=N-SERIAL_THRESHOLD;


    int k, j, start;

// do LU from 0-nlim-1
#pragma omp parallel private(k, j, start)
    {
        int thrid = omp_get_thread_num();
        int nthr = omp_get_num_threads();


        // First touch memory initialization
        for (j = thrid; j < N; j += nthr)
        {
            for (int i = 0; i < N; ++i)
            {
                volatile double temp = A_decompos[i + j * N];
                (void)temp;  
            }
            omp_set_lock(&locks[j]);
        }
#pragma omp barrier

        // First elimination step
        if (thrid == 0)
        {
            const double inv_pivot = 1.0 / A_decompos[0 + 0 * N];
            for (int i = 1; i < N; ++i)
            {
                A_decompos[i + 0 * N] *= inv_pivot;
            }
            omp_unset_lock(&locks[0]);
        }

    

        // k-loop：for each column
        // nlim-loop：for each column
        for (k = 0; k < nlim; k++)
        {

            omp_set_lock(&locks[k]);
            omp_unset_lock(&locks[k]);

            start = (k / nthr) * nthr;
            if (start + thrid <= k)
            {
                start += nthr;
            }

            // update A(k,k) = A(k,k) - A(k,k)*A(k,k)
            for (j = start + thrid; j < N; j += nthr)
            {
                // U-update：A(i,j) = A(i,j) - A(i,k)*A(k,j)
                for (int i = k + 1; i < N; i++)
                {
                    A_decompos[i + j * N] -= A_decompos[i + k * N] * A_decompos[k + j * N];
                }

                // if j == k+1, update L(k+1,k+1) = A(k+1,k+1)/A(k,k)
                if (j == k + 1 && (k + 1) < nlim)
                {
                    const double inv_pivot2 = 1.0 / A_decompos[k + 1 + (k + 1) * N];
                    for (int i = k + 2; i < N; ++i)
                    {
                        A_decompos[i + (k + 1) * N] *= inv_pivot2;
                    }
                    omp_unset_lock(&locks[k + 1]);
                }
            }


            
        }   
    } // end parallel



    //do rest of them
    LU_direct_col(A_decompos, N, nlim);


    // deallocate locks
    for (int i = 0; i < N; i++)
    {
        omp_destroy_lock(&locks[i]);
    }
    free(locks);



}



/**
 * Optimized parallel LU decomposition with column-based locking
 * @para A_decompos: Matrix to decompose (input/output parameter)
 * @para N: Matrix dimension (input parameter)
 */
void show_decomposition(double **A, double **A_decompos, int N)
{
    // Show LU decomposition
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix L + U:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", A_decompos[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    // accept paras: N, thread ,methods:{"serial","naive","optimal"}
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <N> <threads> <method>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // N-matrix size
    int N = atoi(argv[1]);
    if (N <= 0)
    {
        fprintf(stderr, "Error: N must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // threads-number of thread
    int threads = atoi(argv[2]);
    if (threads <= 0)
    {
        fprintf(stderr, "Error: threads must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // method
    char *method = argv[3];
    if (strcmp(method, "serial") != 0 && strcmp(method, "naive") != 0 && strcmp(method, "optimal") != 0)
    {
        fprintf(stderr, "Error: method must be one of {\"serial\", \"naive\", \"optimal\"}.\n");
        return EXIT_FAILURE;
    }



    // print paras
    printf("N = %d, threads = %d, method = %s\n", N, threads, method);

    // set thread number
    omp_set_num_threads(threads);


    double **A;
    A = (double **)malloc(N * sizeof(double *));

    double **A_decompos_row=NULL;
    double *A_decompos=NULL;



    // Part 1: Matrix initialization
    if (strcmp(method, "serial") == 0 || strcmp(method, "naive") == 0)
    {
        A_decompos_row = (double **)malloc(N * sizeof(double *));
        initialize_row(A, A_decompos_row, N);
    }
    else{
        A_decompos = (double *)malloc(N * N * sizeof(double));
        initialize_column(A, A_decompos, N);
    }


    // Part 2: LU decomposition
    double start_time = omp_get_wtime();
    if (strcmp(method, "serial") == 0)
    {
        
        LU_direct(A_decompos_row, N); // Change to LU_direct or LU_naive if needed
    }
    else if (strcmp(method, "naive") == 0)
    {
        
        LU_naive(A_decompos_row, N);
    }
    else
    {
        LU_opt(A_decompos, N);
    }

    double end_time = omp_get_wtime();
    printf("Time taken for LU decomposition: %f seconds\n", end_time - start_time);

    // Part 3: Check correctness of LU decomposition

    int flag=0;
    
    
    if (strcmp(method, "serial") == 0 || strcmp(method, "naive") == 0)
    {
        // Check correctness for row-major order
        flag = check_correctness(A, A_decompos_row, N);
    }
    else
    {
        // Check correctness for column-major order
        flag = check_correctness_col(A, A_decompos, N);
    }
    
    


    if (flag == 1)
    {
        printf("LU decomposition is correct\n");
    }
    else
    {
        printf("LU decomposition is incorrect\n");
        //show_decomposition(A, A_decompos, N);
    }

    // Free memory
    for (int i = 0; i < N; i++)
    {
        free(A[i]);
    }
    free(A);

    if (strcmp(method, "serial") == 0 || strcmp(method, "naive") == 0)
    {
        for (int i = 0; i < N; i++)
        {
            free(A_decompos_row[i]);
        }
        free(A_decompos_row);
    }
    else
    {
    free(A_decompos);
    }
    return 0;
}
