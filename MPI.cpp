// Optimized version of Cannon's Algorithm using MPI
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>

using namespace std;

// Fills a flat matrix with a specific pattern for testing
void fillMatrix(vector<int>& mat, int N, bool pattern = true) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i * N + j] = pattern ? (i + j) : (i * j);
}

// Basic matrix block multiplication: C += A x B
void multiplyBlocks(const vector<int>& A, const vector<int>& B, vector<int>& C, int blockSize) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            for (int k = 0; k < blockSize; ++k)
                C[i * blockSize + j] += A[i * blockSize + k] * B[k * blockSize + j];
}

// Neatly prints a flat matrix with given size
void printMatrix(const vector<int>& mat, int N, const string& name) {
    cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            cout << mat[i * N + j] << " ";
        cout << "\n";
    }
    cout << "\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N; // Size of the full matrix (N x N)
    if (rank == 0) {
        cout << "Enter matrix size (N): ";
        cin >> N;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // Share N with all processes

    int q = sqrt(size); // Processes arranged in a q x q grid
    if (q * q != size || N % q != 0) {
        if (rank == 0)
            cerr << "Error: Number of processes must be a perfect square and N divisible by sqrt(P)." << endl;
        MPI_Finalize();
        return 1;
    }

    int blockSize = N / q;        // Size of each block (submatrix)
    int row = rank / q;           // Process row in processor grid
    int col = rank % q;           // Process column in processor grid

    // Initialize local matrix blocks
    vector<int> A_block(blockSize * blockSize);
    vector<int> B_block(blockSize * blockSize);
    vector<int> C_block(blockSize * blockSize, 0); // Initialize with zeros

    // Temp buffers for shifting
    vector<int> A_temp(blockSize * blockSize);
    vector<int> B_temp(blockSize * blockSize);

    // Static buffers to reuse send data
    static vector<int> A_send(blockSize * blockSize);
    static vector<int> B_send(blockSize * blockSize);

    vector<int> A, B; // Full matrices only held by rank 0
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        fillMatrix(A, N, true);   // Fill A with i+j
        fillMatrix(B, N, false);  // Fill B with i*j

        // Send appropriate blocks to each process
        for (int pr = 0; pr < size; ++pr) {
            int r = pr / q, c = pr % q;
            vector<int> A_sub(blockSize * blockSize), B_sub(blockSize * blockSize);

            // Initial alignment as per Cannon's algorithm
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j) {
                    A_sub[i * blockSize + j] = A[(r * blockSize + i) * N + ((c + q - r) % q * blockSize + j)];
                    B_sub[i * blockSize + j] = B[((r + q - c) % q * blockSize + i) * N + (c * blockSize + j)];
                }

            if (pr == 0) {
                A_block = A_sub;
                B_block = B_sub;
            }
            else {
                MPI_Send(A_sub.data(), blockSize * blockSize, MPI_INT, pr, 0, MPI_COMM_WORLD);
                MPI_Send(B_sub.data(), blockSize * blockSize, MPI_INT, pr, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // Other ranks receive their blocks
        MPI_Recv(A_block.data(), blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block.data(), blockSize * blockSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    double start_time = MPI_Wtime();

    // Main loop of Cannon's algorithm
    for (int step = 0; step < q; ++step) {
        multiplyBlocks(A_block, B_block, C_block, blockSize); // Local block multiply

        // Determine neighbors for circular shift
        int left = (col + q - 1) % q + row * q;
        int right = (col + 1) % q + row * q;
        int up = ((row + q - 1) % q) * q + col;
        int down = ((row + 1) % q) * q + col;

        // Copy blocks into send buffers
        A_send = A_block;
        B_send = B_block;

        // Perform circular shift (left for A, up for B)
        MPI_Sendrecv(A_send.data(), blockSize * blockSize, MPI_INT, left, 0,
            A_temp.data(), blockSize * blockSize, MPI_INT, right, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(B_send.data(), blockSize * blockSize, MPI_INT, up, 1,
            B_temp.data(), blockSize * blockSize, MPI_INT, down, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Load new shifted data into local blocks
        A_block = A_temp;
        B_block = B_temp;
    }

    double end_time = MPI_Wtime();
    double exec_time = end_time - start_time;

    if (rank == 0) {
        // Gather results into full matrix in rank 0
        vector<int> result(N * N);
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j)
                result[i * N + j] = C_block[i * blockSize + j];

        // Collect C blocks from all other processes
        for (int pr = 1; pr < size; ++pr) {
            vector<int> temp(blockSize * blockSize);
            MPI_Recv(temp.data(), blockSize * blockSize, MPI_INT, pr, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int r = pr / q, c = pr % q;
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    result[(r * blockSize + i) * N + (c * blockSize + j)] = temp[i * blockSize + j];
        }

        // Display matrices and performance
        printMatrix(A, N, "Matrix A");
        printMatrix(B, N, "Matrix B");
        printMatrix(result, N, "Result C = A x B");
        cout << "Execution Time: " << exec_time << " seconds" << endl;
    }
    else {
        // Send C block to rank 0
        MPI_Send(C_block.data(), blockSize * blockSize, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
