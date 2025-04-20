#include <iostream>
#include <chrono>
#include "Serial.cpp"
#include "MPI.cpp"
#include "OpenMP.cpp"

// using namespace std;
int main() {
    // Note!! we are sending to the fucntion copies to avoid fault results because it will be passed to three different implemntations
    // however inside each implementation the address of the copy is what is being sent to the the inner functions within each implementation
    // this is to avoid the fault results that might happen if we pass the same address to all three implementations

    int n;

    // Get the size N from the user
    cout << "Enter the size N for the NxN 2D vector: ";
    cin >> n;

    // Input validation: Check if N is a positive integer
    while (cin.fail() || n <= 0) {
        cout << "Invalid input. Please enter a positive integer for N: ";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> n;
    }

    // Initialize the 2D vector of size NxN
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> sum(n, vector<int>(n, 0)); // Initialize sum with zeros

    // Fill the 2D vectors A and B with random values
    srand(time(0)); // Seed for random number generation    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand() % 100; // Random values between 0 and 99
            B[i][j] = rand() % 100; // Random values between 0 and 99
        }
    }
    // Print the original matrices  
    cout << "Matrix A:" << endl;
    print2DVectorGeneric(A);    
    cout << "Matrix B:" << endl;
    print2DVectorGeneric(B);
    cout << "Matrix Sum:" << endl;
    print2DVectorGeneric(sum);
    
    // Call the Serial implementation   
    auto start = chrono::high_resolution_clock::now();
    SerialMain(A, B, sum);      
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout << "Serial implementation time: " << duration << " microseconds" << endl;


    return 0;
}