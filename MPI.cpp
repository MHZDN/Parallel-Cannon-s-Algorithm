#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

// Helper function to print a matrix
void printMatrix(const vector<vector<int>>& mat, const string& name) {
    cout << name << ":\n";
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << "\n";
    }
    cout << "\n";
}

// Helper function to fill a matrix
// If pattern == true, fill with i + j
// If pattern == false, fill with i * j
void fillMatrix(vector<vector<int>>& mat, int N, bool pattern = true) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = pattern ? (i + j) : (i * j);
}

// Multiply two blocks (submatrices) and add the result into C
void multiplyBlocks(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int blockSize) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            for (int k = 0; k < blockSize; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process ID (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    int N; // Matrix size
    if (rank == 0) {
        cout << "Enter matrix size (N): ";
        cin >> N;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast N to all processes

    int q = sqrt(size); // q = number of blocks along a dimension (must be a perfect square)
    if (q * q != size || N % q != 0) {
        if (rank == 0) {
            cerr << "Error: Number of processes must be a perfect square and N must be divisible by sqrt(P)." << endl;
        }
        MPI_Finalize();
        return 1; // Exit if invalid setup
    }

    int blockSize = N / q; // Size of each block
    int row = rank / q;    // Row index of block
    int col = rank % q;    // Column index of block

    // Initialize local matrices
    vector<vector<int>> A_block(blockSize, vector<int>(blockSize)); // Submatrix of A
    vector<vector<int>> B_block(blockSize, vector<int>(blockSize)); // Submatrix of B
    vector<vector<int>> C_block(blockSize, vector<int>(blockSize, 0)); // Result submatrix

    vector<vector<int>> A, B; // Full matrices (only in rank 0)
    if (rank == 0) {
        A.assign(N, vector<int>(N));
        B.assign(N, vector<int>(N));
        fillMatrix(A, N, true);  // Fill A with (i + j)
        fillMatrix(B, N, false); // Fill B with (i * j)

        // Distribute blocks to all processes
        for (int pr = 0; pr < size; ++pr) {
            int r = pr / q;
            int c = pr % q;
            vector<int> A_sub, B_sub;

            // Prepare initial shifted blocks for Cannon's algorithm
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    int a_ij = A[r * blockSize + i][(c + q - r) % q * blockSize + j];
                    int b_ij = B[(r + q - c) % q * blockSize + i][c * blockSize + j];
                    A_sub.push_back(a_ij);
                    B_sub.push_back(b_ij);
                }
            }

            if (pr == 0) {
                // If process 0, copy data directly
                for (int i = 0; i < blockSize; ++i)
                    for (int j = 0; j < blockSize; ++j) {
                        A_block[i][j] = A_sub[i * blockSize + j];
                        B_block[i][j] = B_sub[i * blockSize + j];
                    }
            }
            else {
                // Otherwise, send to other processes
                MPI_Send(A_sub.data(), blockSize * blockSize, MPI_INT, pr, 0, MPI_COMM_WORLD);
                MPI_Send(B_sub.data(), blockSize * blockSize, MPI_INT, pr, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // Receive blocks from process 0
        vector<int> A_recv(blockSize * blockSize), B_recv(blockSize * blockSize);
        MPI_Recv(A_recv.data(), blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_recv.data(), blockSize * blockSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_block[i][j] = A_recv[i * blockSize + j];
                B_block[i][j] = B_recv[i * blockSize + j];
            }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    double start_time = MPI_Wtime(); // Start timing

    // Main Cannon's algorithm
    for (int step = 0; step < q; ++step) {
        multiplyBlocks(A_block, B_block, C_block, blockSize); // Multiply current blocks

        // Calculate neighbor ranks for shifting
        int left = (col + q - 1) % q + row * q;
        int right = (col + 1) % q + row * q;
        int up = ((row + q - 1) % q) * q + col;
        int down = ((row + 1) % q) * q + col;

        // Prepare sending buffers
        vector<int> A_temp(blockSize * blockSize);
        vector<int> B_temp(blockSize * blockSize);
        vector<int> A_send, B_send;
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_send.push_back(A_block[i][j]);
                B_send.push_back(B_block[i][j]);
            }

        // Shift A left and B up
        MPI_Sendrecv(A_send.data(), blockSize * blockSize, MPI_INT, left, 0,
            A_temp.data(), blockSize * blockSize, MPI_INT, right, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(B_send.data(), blockSize * blockSize, MPI_INT, up, 1,
            B_temp.data(), blockSize * blockSize, MPI_INT, down, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Update local blocks after shifting
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_block[i][j] = A_temp[i * blockSize + j];
                B_block[i][j] = B_temp[i * blockSize + j];
            }
    }

    double end_time = MPI_Wtime(); // End timing
    double exec_time = end_time - start_time; // Calculate elapsed time

    if (rank == 0) {
        // Rank 0 gathers the result from all processes
        vector<vector<int>> result(N, vector<int>(N));

        // Copy own computed block
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j)
                result[i][j] = C_block[i][j];

        // Receive C_blocks from other processes
        for (int pr = 1; pr < size; ++pr) {
            vector<int> temp(blockSize * blockSize);
            MPI_Recv(temp.data(), blockSize * blockSize, MPI_INT, pr, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int r = pr / q;
            int c = pr % q;
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    result[r * blockSize + i][c * blockSize + j] = temp[i * blockSize + j];
        }

        // Print matrices A, B, and C
        printMatrix(A, "Matrix A");
        printMatrix(B, "Matrix B");
        cout << "Result C = A x B:" << endl;
        printMatrix(result, "C");
        cout << "Execution Time: " << exec_time << " seconds" << endl;
    }
    else {
        // Other processes send their C_blocks to rank 0
        vector<int> C_send;
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j)
                C_send.push_back(C_block[i][j]);
        MPI_Send(C_send.data(), blockSize * blockSize, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
