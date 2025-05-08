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

// Copy a block from source to destination
void copyBlock(const vector<vector<int>>& src, vector<vector<int>>& dst, int srcRow, int srcCol, int dstRow, int dstCol, int blockSize) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            dst[dstRow + i][dstCol + j] = src[srcRow + i][srcCol + j];
        }
    }
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

    // Fill A and B with test values
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j;
        }

    printMatrix(A, "Initial A");
    printMatrix(B, "Initial B");

    double start_time = omp_get_wtime();

    if (blockSize == 1) {
        cout << "\nNote: blockSize = 1. Falling back to standard matrix multiplication.\n";
#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    else {
        // Initial skewing using block-level shift
        vector<vector<int>> A_skewed(N, vector<int>(N));
        vector<vector<int>> B_skewed(N, vector<int>(N));

        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                int srcRow = i * blockSize;
                int srcCol = j * blockSize;

                int dstACol = ((j + q - i) % q) * blockSize;
                int dstBRow = ((i + q - j) % q) * blockSize;

                copyBlock(A, A_skewed, srcRow, srcCol, srcRow, dstACol, blockSize);
                copyBlock(B, B_skewed, srcRow, srcCol, dstBRow, srcCol, blockSize);
            }
        }

        A = A_skewed;
        B = B_skewed;

        // Main computation loop
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
                                sum += A[rowStart + i][(bj * blockSize + k)] *
                                    B[(bi * blockSize + k)][colStart + j];
                            }
                            C[rowStart + i][colStart + j] += sum;
                        }
                    }
                }
            }

            // Rotate A blocks left and B blocks up
            vector<vector<int>> A_rotated(N, vector<int>(N));
            vector<vector<int>> B_rotated(N, vector<int>(N));

            for (int i = 0; i < q; i++) {
                for (int j = 0; j < q; j++) {
                    int srcRow = i * blockSize;
                    int srcCol = j * blockSize;

                    int dstACol = ((j + q - 1) % q) * blockSize;
                    int dstBRow = ((i + q - 1) % q) * blockSize;

                    copyBlock(A, A_rotated, srcRow, srcCol, srcRow, dstACol, blockSize);
                    copyBlock(B, B_rotated, srcRow, srcCol, dstBRow, srcCol, blockSize);
                }
            }

            A = A_rotated;
            B = B_rotated;
        }
    }

    double end_time = omp_get_wtime();
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;

    printMatrix(C, "Result C = A x B");
    return 0;
}
