#include <iostream>
#include<vector>
using namespace std;

// incase we want to do for float or int or double 
template<typename T>
void print2DVectorGeneric(const vector<vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (const T& val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

template<typename T>
void leftShiftRow(vector<T>&matrix, int shiftTimes)
{
    for(int i=0 ; i<shiftTimes ; i++)
    {
        T temp = matrix[0];
        for(int j=0 ; j<matrix.size()-1 ; j++)
        {
            matrix[j] = matrix[j+1];
        }
        matrix[matrix.size()-1] = temp;
    }
    
}

template<typename T>
void upShiftColumn(vector<vector<T>>&matrix, int shiftTimes, int column_no)
{
    for(int i=0 ; i<shiftTimes ; i++)
    {
        T temp = matrix[0][column_no];
        for(int j=0 ; j<matrix.size()-1 ; j++)
        {
            matrix[j][column_no] = matrix[j+1][column_no];
        }
        matrix[matrix.size()-1][column_no] = temp;
    }
}

template<typename T>
void pairWiseMultiplication(const vector<vector<T>>&A,const vector<vector<T>>&B, vector<vector<T>>&sum)
{
    for(int i=0 ; i<A.size() ; i++)
    {
        for(int j=0 ; j<A[i].size() ; j++)
        {
            sum[i][j] += A[i][j] * B[i][j];
        }
    }
}

template<typename T>
void SerialMain(vector<vector<T>> arr1,vector<vector<T>> arr2, vector<vector<T>> sum){


    for(int i = 0; i < arr1.size(); i++){

        if(i==0)
        {
            // initialization shift by the index of row/column 
            // j represents the row/column of the array 
            // j here also represents the no. of shifts 
            for(int j = 1; j < arr1.size(); j++)
            {
                // shifting each row of arr1 
                leftShiftRow(arr1[j],j);

                // shifting each column in arr2
                // noticing that we are shifting column which is element inside each inner vector 
                // so we are passing the whole vector
                upShiftColumn(arr2,j,j);
            }
        }
        else
        {
            // 'j' represents the row/column of the array
            for(int j = 0; j < arr1.size(); j++)
            {
                leftShiftRow(arr1[j],1);
                upShiftColumn(arr2,1,j);
            }
        }

        pairWiseMultiplication(arr1,arr2,sum);

    }
    
    cout << "Matrix sum After Serial Cannon:" << endl;
    print2DVectorGeneric(sum);


}




