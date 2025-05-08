#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cassert>

using namespace std;

int N = 3; // Default matrix size

// Utility: Print matrix
void printMatrix(const vector<vector<int>>& mat, const string& name) {
    cout << name << ":\n";
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << "\n";
    }
    cout << "\n";
}

// Shift row left by blockSize (circular)
void shiftRowLeft(vector<vector<int>>& mat, int row, int blockSize) {
    vector<int> temp = mat[row];
    for (int i = 0; i < N; i++) {
        mat[row][i] = temp[(i + blockSize) % N];
    }
}

// Shift column up by blockSize (circular)
void shiftColUp(vector<vector<int>>& mat, int col, int blockSize) {
    vector<int> temp(N);
    for (int i = 0; i < N; i++)
        temp[i] = mat[i][col];
    for (int i = 0; i < N; i++)
        mat[i][col] = temp[(i + blockSize) % N];
}

int main() {
    cout << "Enter matrix size (N): ";
    cin >> N;

    int num_threads;
    cout << "Enter number of threads (perfect square): ";
    cin >> num_threads;
    omp_set_num_threads(num_threads);

    double root = sqrt(num_threads);
    if (floor(root) != root) {
        cerr << "Error: Number of threads must be a perfect square." << endl;
        return 1;
    }

    int q = static_cast<int>(root);
    if (N % q != 0) {
        cerr << "Error: Matrix size must be divisible by sqrt(num_threads)." << endl;
        return 1;
    }

    int blockSize = N / q;

    // Initialize matrices A, B, and C
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));

    // Fill A and B with test values (e.g., A[i][j] = i + j, B[i][j] = i * j)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j;
        }

    printMatrix(A, "Initial A");
    printMatrix(B, "Initial B");

    // Initial skewing
#pragma omp parallel for collapse(2)
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            shiftRowLeft(A, i * blockSize, j);
            shiftColUp(B, j * blockSize, i);
        }
    }

    double start_time = omp_get_wtime();

    // Main computation loop (q steps)
    for (int step = 0; step < q; step++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int bi = 0; bi < q; bi++) {
            for (int bj = 0; bj < q; bj++) {
                int rowStart = bi * blockSize;
                int colStart = bj * blockSize;
                for (int i = 0; i < blockSize; i++) {
                    for (int j = 0; j < blockSize; j++) {
                        int sum = 0;
                        for (int k = 0; k < blockSize; k++) {
                            sum += A[rowStart + i][(bj * blockSize + k) % N] *
                                B[(bi * blockSize + k) % N][colStart + j];
                        }
                        C[rowStart + i][colStart + j] += sum;
                    }
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < q; i++) {
            shiftRowLeft(A, i * blockSize, 1);
            shiftColUp(B, i * blockSize, 1);
        }
    }

    double end_time = omp_get_wtime();
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;

    printMatrix(C, "Result C = A x B");
    return 0;
}