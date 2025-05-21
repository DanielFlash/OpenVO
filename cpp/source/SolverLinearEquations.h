/*Copyright (c) <2024> <OOO "ORIS">
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.*/
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

/// <summary>
/// Class of SLAE solution using SVD algorithm
/// </summary>
class SvdSolver
{
private:
    double COS, SIN; //Variables for accumulation of sine and cosine of rotation
    #define A_p(i,j) A_ptr[(i)*mi + (j)]
    #define P_p(i,j) P_ptr[(i)*ma + (j)]
    #define Q_p(i,j) Q_ptr[(i)*mi + (j)]

    double prepareParamRotate(double x, double y) 
    {
        //Calculating COS and SIN for a planar rotation:
        double r = sqrtl(x * x + y * y);
        if (x < 0.0L)
            r = -r; //The sign of r corresponds to the sign of x

        if (r == 0.0L) 
        {
            COS = 1.0L;
            SIN = 0.0L;
        }
        else 
        {
            COS = x / r;
            SIN = y / r;
        }

        return r;
    }

    /// <summary>
    /// Performs n consecutive planar rotations
    /// </summary>
    /// <param name="n"></param>
    /// <param name="x1"></param>
    /// <param name="dx"></param>
    /// <param name="y1"></param>
    /// <param name="dy"></param>
    void rotateCellMatrix(int n, double* x1, int dx, double* y1, int dy) 
    {
        double u;
        double v;
        for (int i = 0; i < n; i++) 
        {
            u = *x1;
            v = *y1;
            *x1 = COS * u + SIN * v;
            *y1 = -SIN * u + COS * v;
            x1 += dx;
            y1 += dy;
        }
    }

    /// <summary>
    /// Singular value decomposition method
    /// </summary>
    /// <param name="m">Number of matrix rows</param>
    /// <param name="n">Number of matrix columns</param>
    /// <param name="withU">Whether to accumulate the lower part of the output matrix</param>
    /// <param name="a">Flat matrix of the original equation</param>
    /// <param name="u">Complex unitary matrix</param>
    /// <param name="v">Complex unitary matrix</param>
    /// <param name="d"></param>
    /// <param name="e"></param>
    /// <returns>Number of QR iterations with implicit shift</returns>
    int SVD(int m, int n, int withU, vector<double> &a, vector<double> &u, 
        vector<double> &v, vector<double> &d, vector<double> &e) 
    {
        int ma, mi;
        int withP; //Flags for accumulation of rotation matrices
        int i, j, k, ell;
        double bulge; // bulge

        double eps = 1e-12;//We set the machine epsilon
        double tol; //Threshold for zero check
        double sigma; //Shift
        double x, y, z, f, g, h;

        //Determine which side is larger; In our case, always m >= n
        //Rotation matrices P and Q so that in the end A = P'*D*Q' (if m >= n)
        withP = withU;
        ma = m;
        mi = n;

        //Allocating memory for internal variables using vectors
        std::vector<double> A_storage(m * n);
        double* A_ptr = A_storage.data();
        std::vector<double> D_storage(mi);
        std::vector<double> E_storage(mi);
        double* D_ptr = D_storage.data();
        double* E_ptr = E_storage.data();
        std::vector<double> P_storage;
        double* P_ptr = nullptr;
        if (withP) 
        {
            P_storage.resize(ma * ma, 0.0L);
            P_ptr = P_storage.data();
        }

        std::vector<double> Q_storage;
        double* Q_ptr = nullptr;
        Q_storage.resize(mi * mi, 0.0L);
        Q_ptr = Q_storage.data();
        std::vector<int> index(ma);

        //Copy the input matrix a to the internal array A_ptr.
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                A_p(i, j) = a[(i)*mi + (j)];
            }
        }

        //Initialization of matrices P and Q to unity
        if (withP) 
        {
            for (i = 0; i < ma * ma; i++)
            {
                P_ptr[i] = 0.0L;
            }

            for (i = 0; i < ma; i++)
            {
                P_ptr[i * ma + i] = 1.0L;
            }
        }
        
        for (i = 0; i < mi * mi; i++)
        {
            Q_ptr[i] = 0.0L;
        }

        for (i = 0; i < mi; i++)
        {
            Q_ptr[i * mi + i] = 1.0L;
        }

        //Step 1. Bidigationalization
        for (j = 0; j < mi; j++) 
        {
            //Left turns. We try to zero out the elements A(i,j), i = j+1,...,ma-1.
            for (i = j + 1; i < ma; i++) 
            {
                if (A_p(i, j) != 0.0L) 
                { 
                    prepareParamRotate(A_p(j, j), A_p(i, j));
                    rotateCellMatrix(mi - j, &A_ptr[j * mi + j], 1, &A_ptr[i * mi + j], 1);
                    if (withP)
                        rotateCellMatrix(ma, &P_ptr[j * ma + 0], 1, &P_ptr[i * ma + 0], 1);
                }
            }
            D_ptr[j] = A_p(j, j);

            // Right turns. We try to zero out the elements A(j,k), k = j+2,...,mi-1.
            for (k = j + 2; k < mi; k++) 
            {
                if (A_p(j, k) != 0.0L) 
                {
                    prepareParamRotate(A_p(j, j + 1), A_p(j, k));
                    rotateCellMatrix(ma - j, &A_ptr[j * mi + j + 1], mi, &A_ptr[j * mi + k], mi);
                    rotateCellMatrix(mi, &Q_ptr[0 * mi + j + 1], mi, &Q_ptr[0 * mi + k], mi);
                        
                }
            }

            if (j < mi - 1)
                E_ptr[j + 1] = A_p(j, j + 1);
        }

        E_ptr[0] = 0.0L;
        // Calculate the threshold for checking for zero elements
        tol = fabsl(D_ptr[mi - 1]);
        for (i = 0; i < mi - 1; i++) 
        {
            x = fabsl(D_ptr[i]) + fabsl(E_ptr[i + 1]);
            if (x > tol)
                tol = x;
        }

        tol *= eps;
        //Step 2. QR iteration with implicit shift
        int qrSteps = 0; //QR iteration counter
        //Outer loop on columns
        for (k = mi - 1; k >= 0; k--) 
        {
            while (true) 
            {
                //We determine the initial index of the ell block. If a split point is found (fabsl(E[ell]) <= tol), we break the loop.
                bool splitFound = false;
                for (ell = k; ell >= 0; ell--) 
                {
                    if (fabsl(E_ptr[ell]) <= tol) 
                    {
                        splitFound = true;
                        break; // found the split point
                    }
                    if ((ell - 1) >= 0 && fabsl(D_ptr[ell - 1]) <= tol) 
                    {
                        break; //"cancellation" clause
                    }
                }
                
                //If the cancellation condition is met (i.e. no split found)
                if (!splitFound && (ell - 1) >= 0 && fabsl(D_ptr[ell - 1]) <= tol) 
                {
                    COS = 0.0L;
                    SIN = 1.0L;
                    for (i = ell; i <= k; i++) 
                    {
                        bulge = SIN * E_ptr[i];
                        E_ptr[i] = COS * E_ptr[i];
                        if (fabsl(bulge) <= tol)
                            break;

                        D_ptr[i] = prepareParamRotate(D_ptr[i], -bulge);
                        if (withP)
                            rotateCellMatrix(ma, &P_ptr[(ell - 1) * ma + 0], 1, &P_ptr[i * ma + 0], 1);
                    }
                }

                z = D_ptr[k];
                //If the partition is found (ell == k) – convergence is achieved
                if (ell == k)
                    break; //exit the while loop for a given k

                qrSteps++;
                x = D_ptr[ell];
                if (0/*useRayleighShift*/) //Rayleigh shift. Not used yet. Slower than Wilkinson
                {
                    sigma = D_ptr[k] * D_ptr[k] + E_ptr[k] * E_ptr[k];
                    f = x - sigma / x;
                }
                else //Wilkinson's shift
                {
                    if (qrSteps <= 1) 
                    {
                        sigma = 0.0L;
                        f = x;
                    }
                    else 
                    {
                        y = D_ptr[k - 1];
                        g = E_ptr[k - 1];
                        h = E_ptr[k];
                        f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
                        g = sqrtl(f * f + 1.0L);
                        if (f < 0.0L)
                            g = -g;
                        sigma = z * z + h * (h - y / (f + g));
                        f = ((x - z) * (x + z) + (y / (f + g) - h) * h) / x;
                    }
                }

                COS = SIN = 1.0L;
                for (i = ell + 1; i <= k; i++) 
                {
                    g = E_ptr[i];
                    y = D_ptr[i];
                    h = SIN * g;
                    g = COS * g;
                    E_ptr[i - 1] = prepareParamRotate(f, h);
                    f = x * COS + g * SIN;
                    g = -x * SIN + g * COS;
                    h = y * SIN;
                    y = y * COS;
                    rotateCellMatrix(mi, &Q_ptr[0 * mi + (i - 1)], mi, &Q_ptr[0 * mi + i], mi);

                    D_ptr[i - 1] = prepareParamRotate(f, h);
                    f = g * COS + y * SIN;
                    x = -g * SIN + y * COS;
                    if (withP)
                        rotateCellMatrix(ma, &P_ptr[(i - 1) * ma + 0], 1, &P_ptr[i * ma + 0], 1);
                }

                E_ptr[ell] = 0.0L;
                E_ptr[k] = f;
                D_ptr[k] = x;
                //After each QR step, the while loop is repeated to check for convergence
            }

            if (z < 0.0L) 
            {
                D_ptr[k] = -z;
                for (i = 0; i < mi; i++)
                {
                    Q_ptr[i * mi + k] = -Q_ptr[i * mi + k];
                }
            }

            E_ptr[k] = qrSteps; //We save the number of iterations for the k-th step
        }

        //Adjusting QR iteration counters
        k = 0;
        ell = qrSteps;
        for (i = mi - 1; i >= 0; i--) 
        {
            j = (int)E_ptr[i];
            E_ptr[i] = (mi - i) + 0.001L * (j - k);
            k = j;
        }

        //Sorting singular values ​​in descending order
        for (i = 0; i < mi; i++) 
        {
            x = 0.0L;
            for (j = 0; j < mi; j++) 
            {
                if (D_ptr[j] >= x) 
                {
                    x = D_ptr[j];
                    k = j;
                }
            }

            index[i] = k; //Index of the i-th largest value
            D_ptr[k] = -D_ptr[k] - 1.0L; //Let's make it so that you don't have to select it again
        }
        //Recovering True Singular Values
        for (i = 0; i < mi; i++)
        {
            D_ptr[i] = -D_ptr[i] - 1.0L;
        }
        //The remaining indices remain unchanged.
        for (i = mi; i < ma; i++)
        {
            index[i] = i;
        }

        //Copying results to external vectors
        for (i = 0; i < mi; i++) 
        {
            d[i] = (double)D_ptr[index[i]];
            e[i] = (double)E_ptr[index[i]];
        }

        if (withU) 
        {
            //U = P'
            for (j = 0; j < m; j++)
            {
                for (i = 0; i < m; i++)
                {
                    u[j * m + i] = (double)P_ptr[index[i] * ma + j];
                }
            }
        }

        //V = Q
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                v[i * n + j] = (double)Q_ptr[i * mi + index[j]];
            }
        }

        return ell;
    }

public:
    /// <summary>
    /// Solution of SLAE for homogeneous formulation A*h=0
    /// </summary>
    /// <param name="rows">Number of rows in equation</param>
    /// <param name="cols">Number of columns in equation</param>
    /// <param name="A">Matrix in one row of equation A*h</param>
    /// <returns>Vector of SLAE solution</returns>
    std::vector<double> solveHomogeneous(size_t rows, size_t cols, std::vector<double> A)
    {
        std::vector<double> U(rows * rows);
        std::vector<double> V(cols * cols);
        std::vector<double> D(std::min(rows, cols));
        std::vector<double> E(std::min(rows, cols));

        int qrSteps = SVD(rows, cols, 0, A, U, V, D, E);

        //Extract the column from V that corresponds to the minimum singular value (ALREADY SORTED)
        vector<double> x(9);
        for (int i = 0; i < 9; ++i)
        {
            x[i] = V[i * 9 + 8];
        }

        return x;
    }

    /// <summary>
    /// Solution of SLAE via direct linear equation A*p = b
    /// </summary>
    /// <param name="rows">Number of rows in matrix A</param>
    /// <param name="cols">Number of columns in matrix A</param>
    /// <param name="A_">Matrix of equation A*p</param>
    /// <returns>Vector of SLAE solution result</returns>
    std::vector<double> solveDirectLinear(size_t rows, size_t cols, 
        pair< std::vector<std::vector<double>>, vector<double>> A_)
    {
        std::vector<double> A(rows * cols);
        size_t iter = 0;
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                A[iter] = A_.first[i][j];
                iter++;
            }
        }

        std::vector<double> U(rows * rows);
        std::vector<double> V(cols * cols);
        std::vector<double> d(std::min(rows, cols));
        std::vector<double> e(std::min(rows, cols));

        int qrSteps = SVD(rows, cols, 1, A, U, V, d, e);

        //Now we need to extract the solution via the pseudo-inverse
        //1) Calculate U^T*b
        std::vector<double> tmp(rows, 0.0);
        for (int i = 0; i < rows; i++) 
        {
            double sum = 0.0;
            for (int row = 0; row < rows; row++) 
            {
                sum += U[row * rows + i] * A_.second[row];
            }

            tmp[i] = sum;
        }

        //2) We divide the first k components by singular values
        for (int i = 0; i < cols; i++) 
        {
            if (std::fabs(d[i]) < 1.0e-12) 
            {
                // Case of degeneration
                cout << "Error in calculating the result of solving SLAE. Degeneration case";
                return std::vector<double>();
                //tmp[i] = 0.0;
            }
            else 
            {
                tmp[i] /= d[i];
            }
        }

        //3) p = V * w
        std::vector<double> p(cols, 0.0);
        for (int row = 0; row < cols; row++) 
        {
            double sum = 0.0;
            for (int col = 0; col < cols; col++) 
            {
                sum += V[row * cols + col] * tmp[col];
            }

            p[row] = sum;
        }

        return p;
    }
};

/// <summary>
/// Class of SLAE solution
/// </summary>
class LuSolver
{
private:
    double eps = 1e-12;//We set the machine epsilon

public:
    /// <summary>
    /// Method of solving SLAE
    /// Format: A*X = b
    /// </summary>
    /// <param name="A">Left side of equation A*X</param>
    /// <param name="b">Right side of equation b</param>
    /// <returns>Vector X is the solution of SLAE. Zero if no solution is found</returns>
    vector<double> solve(vector<vector<double>> A,  vector<double> b)
    {
        bool resultStatus = false;
        vector<double> x;
        vector<vector<double>> L, U;
        vector<int> P, Q;
        resultStatus = FullPivotLUDecomposition(A, L, U, P, Q);
        if (!resultStatus)
        {
            cout << "SOLVER ERROR\n";
            return vector<double>();
        }
        
        resultStatus = SolveLinearSystem(L, U, P, Q, b, x);
        if (!resultStatus)
        {
            cout << "SOLVER ERROR2\n";
            return vector<double>();
        }

        return x;
    }

    /// <summary>
    /// Function for LU decomposition with full pivot selection
    /// </summary>
    /// <param name="A">Original matrix</param>
    /// <param name="L">Lower triangular matrix</param>
    /// <param name="U">Upper triangular matrix</param>
    /// <param name="P">Permutation matrix</param>
    /// <param name="Q">Permutation matrix</param>
    /// <returns>Result of the solution</returns>
    bool FullPivotLUDecomposition(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, vector<int>& P, vector<int>& Q)
    {
        int n = A.size();
        if (n == 0 || A[0].size() != n)
        {
            cout << "Matrix A must be square and of non-zero size." << std::endl;
            return false;
        }

        U = A;
        L = vector<vector<double>>(n, vector<double>(n, 0.0));
        P.resize(n);
        Q.resize(n);
        for (int i = 0; i < n; ++i)
        {
            P[i] = i;
            Q[i] = i;
        }

        for (int k = 0; k < n; ++k)
        {
            //Search for a reference element
            double max = 0.0;
            int p = k, q = k;
            for (int i = k; i < n; ++i)
            {
                for (int j = k; j < n; ++j)
                {
                    if (fabs(U[i][j]) > max)
                    {
                        max = fabs(U[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }
            if (max < eps)
            {
                cout << "The matrix is ​​degenerate or ill-conditioned" << std::endl;
                return false;
            }
            //Permutation of rows k and p in U
            swap(U[k], U[p]);
            swap(P[k], P[p]);

            //Permutation of columns k and q in U
            for (int i = 0; i < n; ++i)
            {
                swap(U[i][k], U[i][q]);
            }

            swap(Q[k], Q[q]);

            //Checking for division by zero
            if (fabs(U[k][k]) < eps)
            {
                cout << "Zero or near zero support element." << std::endl;
                return false;
            }

            //Updating the U matrix and calculating the factors
            for (int i = k + 1; i < n; ++i)
            {
                U[i][k] /= U[k][k];
                for (int j = k + 1; j < n; ++j)
                {
                    U[i][j] -= U[i][k] * U[k][j];
                }
            }
        }

        //Extraction of matrices L and U
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i > j)
                    L[i][j] = U[i][j];
                else if (i == j)
                    L[i][j] = 1.0;
                else
                    L[i][j] = 0.0;

                if (i <= j)
                    U[i][j] = U[i][j];
                else
                    U[i][j] = 0.0;
            }
        }

        return true;
    }

    /// <summary>
    /// Function for solving a system of linear equations
    /// </summary>
    /// <param name="L">Lower triangular matrix</param>
    /// <param name="U">Upper triangular matrix</param>
    /// <param name="P">Permutation matrix</param>
    /// <param name="Q">Permutation matrix</param>
    /// <param name="b">Right side of the equation</param>
    /// <param name="x">Left side of the equation</param>
    /// <returns>Result of the solution</returns>
    bool SolveLinearSystem(const vector<vector<double>>& L, const vector<vector<double>>& U, const vector<int>& P, const vector<int>& Q, const vector<double>& b, vector<double>& x)
    {
        int n = L.size();
        if (b.size() != n)
        {
            cout << "The size of the vector b does not match the size of the matrix." << std::endl;
            return false;
            //throw invalid_argument("");
        }

        vector<double> Pb(n);
        //Applying a row permutation to a vector b
        for (int i = 0; i < n; ++i)
        {
            Pb[i] = b[P[i]];
        }

        //Direct move (solution Ly = Pb)
        vector<double> y(n);
        for (int i = 0; i < n; ++i)
        {
            y[i] = Pb[i];
            for (int j = 0; j < i; ++j)
            {
                y[i] -= L[i][j] * y[j];
            }
        }

        //Reverse move (solution Ux = y)
        vector<double> tempX(n);
        for (int i = n - 1; i >= 0; --i)
        {
            tempX[i] = y[i];
            for (int j = i + 1; j < n; ++j)
            {
                tempX[i] -= U[i][j] * tempX[j];
            }

            if (fabs(U[i][i]) < eps)
            {
                cout << "Zero or near zero element on diagonal U." << std::endl;
                return false;
            }

            tempX[i] /= U[i][i];
        }

        //Applying a column permutation to a vector x
        x.resize(n);
        for (int i = 0; i < n; ++i)
        {
            x[Q[i]] = tempX[i];
        }

        return true;
    }
};
