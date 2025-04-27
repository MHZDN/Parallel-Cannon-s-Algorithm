#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

void printMatrix(const vector<vector<int>>& mat, const string& name) {
    cout << name << ":\n";
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << "\n";
    }
    cout << "\n";
}

void fillMatrix(vector<vector<int>>& mat, int N, bool pattern = true) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = pattern ? (i + j) : (i * j);
}

void multiplyBlocks(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int blockSize) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            for (int k = 0; k < blockSize; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        cout << "Enter matrix size (N): ";
        cin >> N;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q = sqrt(size);
    if (q * q != size || N % q != 0) {
        if (rank == 0) {
            cerr << "Error: Number of processes must be a perfect square and N must be divisible by sqrt(P)." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int blockSize = N / q;
    int row = rank / q;
    int col = rank % q;

    vector<vector<int>> A_block(blockSize, vector<int>(blockSize));
    vector<vector<int>> B_block(blockSize, vector<int>(blockSize));
    vector<vector<int>> C_block(blockSize, vector<int>(blockSize, 0));

    vector<vector<int>> A, B;
    if (rank == 0) {
        A.assign(N, vector<int>(N));
        B.assign(N, vector<int>(N));
        fillMatrix(A, N, true);
        fillMatrix(B, N, false);

        for (int pr = 0; pr < size; ++pr) {
            int r = pr / q;
            int c = pr % q;
            vector<int> A_sub, B_sub;
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    int a_ij = A[r * blockSize + i][(c + q - r) % q * blockSize + j];
                    int b_ij = B[(r + q - c) % q * blockSize + i][c * blockSize + j];
                    A_sub.push_back(a_ij);
                    B_sub.push_back(b_ij);
                }
            }
            if (pr == 0) {
                for (int i = 0; i < blockSize; ++i)
                    for (int j = 0; j < blockSize; ++j) {
                        A_block[i][j] = A_sub[i * blockSize + j];
                        B_block[i][j] = B_sub[i * blockSize + j];
                    }
            }
            else {
                MPI_Send(A_sub.data(), blockSize * blockSize, MPI_INT, pr, 0, MPI_COMM_WORLD);
                MPI_Send(B_sub.data(), blockSize * blockSize, MPI_INT, pr, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        vector<int> A_recv(blockSize * blockSize), B_recv(blockSize * blockSize);
        MPI_Recv(A_recv.data(), blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_recv.data(), blockSize * blockSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_block[i][j] = A_recv[i * blockSize + j];
                B_block[i][j] = B_recv[i * blockSize + j];
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < q; ++step) {
        multiplyBlocks(A_block, B_block, C_block, blockSize);

        int left = (col + q - 1) % q + row * q;
        int right = (col + 1) % q + row * q;
        int up = ((row + q - 1) % q) * q + col;
        int down = ((row + 1) % q) * q + col;

        vector<int> A_temp(blockSize * blockSize);
        vector<int> B_temp(blockSize * blockSize);
        vector<int> A_send, B_send;
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_send.push_back(A_block[i][j]);
                B_send.push_back(B_block[i][j]);
            }

        MPI_Sendrecv(A_send.data(), blockSize * blockSize, MPI_INT, left, 0,
            A_temp.data(), blockSize * blockSize, MPI_INT, right, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(B_send.data(), blockSize * blockSize, MPI_INT, up, 1,
            B_temp.data(), blockSize * blockSize, MPI_INT, down, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j) {
                A_block[i][j] = A_temp[i * blockSize + j];
                B_block[i][j] = B_temp[i * blockSize + j];
            }
    }

    double end_time = MPI_Wtime();
    double exec_time = end_time - start_time;

    if (rank == 0) {
        vector<vector<int>> result(N, vector<int>(N));
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j)
                result[i][j] = C_block[i][j];

        for (int pr = 1; pr < size; ++pr) {
            vector<int> temp(blockSize * blockSize);
            MPI_Recv(temp.data(), blockSize * blockSize, MPI_INT, pr, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int r = pr / q;
            int c = pr % q;
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    result[r * blockSize + i][c * blockSize + j] = temp[i * blockSize + j];
        }

        cout << "Result C = A x B:" << endl;
        printMatrix(result, "C");
        cout << "Execution Time: " << exec_time << " seconds" << endl;
    }
    else {
        vector<int> C_send;
        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; ++j)
                C_send.push_back(C_block[i][j]);
        MPI_Send(C_send.data(), blockSize * blockSize, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
