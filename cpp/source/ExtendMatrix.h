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
#include <valarray>

template <typename T = double>
class Vector
{
private:
    std::valarray<double> data;

public:
    Vector() {};

    Vector(size_t size)
    {
        data.resize(size);
    }

    Vector(size_t size, T initVal)
    {
        data.resize(size);
        for (size_t i = 0; i < size; i++)
        {
            data[i] = initVal;
        }
    }

    void initValue(T value)
    {
        data = value;
    }

    void flush()
    {
        resize(0);
    }

    void resize(size_t size)
    {
        data.resize(size);
    }

    void scale(double factor)
    {
        data *= factor;
    }

    double max()
    {
        return data.max();
    }

    double sum()
    {
        return data.sum();
    }

    size_t size()
    {
        return data.size();
    }

    double& operator[] (size_t i)
    {
        return data[i];
    }

    const double& operator[](size_t i) const
    {
        return data[i];
    }

    void operator= (std::valarray<double> B)
    {
        data = B;
    }

    Vector operator* (double& value)
    {
        Vector C(size());

        C = data * value;
        return C;
    }

    Vector operator* (std::valarray<double>& B)
    {
        Vector result;
        result = data * B;
        return result;
    }

    Vector operator* (Vector& B)
    {
        return B * data;
    }

    friend Vector operator*(double& value, Vector& vector)
    {
        return vector * value;
    }

    Vector& operator *=(const double& multiplue)
    {
        data *= multiplue;
        return *this;
    }

    Vector operator / (const Vector& del)
    {
        Vector result(data.size());

        for (int i = 0; i < data.size(); i++)
        {
            result[i] = data[i] / del[i];
        }

        return result;
    }
};

template <typename T = double>
class Matrix
{
private:
    std::valarray<Vector<T>> data;
public:
    Matrix()
    {

    }

    Matrix(size_t row, size_t column)
    {
        resize(row, column);
    }

    Matrix(size_t row, size_t column, T value)
    {
        resize(row, column);
        initValue(value);
    }

    Matrix(size_t* size)
    {
        resize(size[0], size[1]);
    }

    void initValue(T value)
    {
        if (sizeColumn() == 0)
            return;

        for (size_t i = 0; i < sizeRow(); i++)
        {
            data[i].initValue(value);
        }
    }

    ~Matrix()
    {
        resize(0, 0);
    }

    void flush()
    {
        resize(0, 0);
    }

    double get(int number)
    {
        int row = number / sizeRow();
        int column = number % sizeRow();

        return data[row][column];
    }

    Matrix operator* (double& value)
    {
        Matrix C(sizeRow(), sizeColumn());

        for (int i = 0; i < sizeRow(); i++)
        {
            C[i] = value * data[i];
        }

        return C;
    }

    Vector<T> operator*(Vector<T>& vector)
    {
        Vector<T> result(vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            result[i] = (data[i] * vector).sum();
        }

        return result;
    }

    Matrix operator* (Matrix& B)
    {
        Matrix C(sizeRow(), sizeColumn());
        double summ;

        for (size_t i = 0; i < sizeRow(); i++)
        {
            for (size_t j = 0; j < sizeColumn(); j++)
            {
                summ = 0;
                for (size_t k = 0; k < sizeColumn(); k++)
                {
                    summ += data[i][k] * B[k][j];
                }

                C[i][j] = summ;
            }
        }

        return C;
    }

    double& operator()(int r, int c)
    {
        return data[r][c];
    }

    double operator()(int r, int c) const
    {
        return data[r][c];
    }

    Vector<T>& operator[] (size_t row)
    {
        return data[row];
    }

    const Vector<T>& operator[](size_t row) const
    {
        return data[row];
    }

    void resize(size_t row, size_t column)
    {
        data.resize(row);

        for (int i = 0; i < row; i++)
        {
            data[i] = Vector<T>(column);
        }
    }

    void resize(size_t row, size_t column, T initVal)
    {
        data.resize(row);

        for (int i = 0; i < row; i++)
        {
            data[i] = Vector<T>(column);
            data[i] = initVal;
        }
    }

    Vector<T> getVector(size_t numberRow)
    {
        Vector row;
        row = data[numberRow];

        return row;
    }

    void setVector(Vector<T> row, int numberRow)
    {
        data[numberRow] = row;
    }

    void resizeColumn(size_t numberRow, size_t count)
    {
        data[numberRow].resize(count);
    }

    void resizeRow(size_t count)
    {
        data.resize(count);
    }

    size_t sizeRow()
    {
        return data.size();
    }

    size_t sizeColumn()
    {
        if (data.size() == 0)
            return 0;

        return data[0].size();
    }

    void identity()
    {
        size_t rows = sizeRow(), cols = sizeColumn();
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                data[i][j] = 0.0;
            }
            data[i][i] = 1.0;
        }
    }
};

template <typename T = double>
class Tensor
{
private:
    std::valarray<Matrix<T>> data;
public:
    Tensor();

    Tensor(size_t row, size_t column, size_t tube)
    {
        resize(row, column, tube);
    }

    void exportImage(Matrix<T> inp)
    {
        resize(inp.sizeRow(), inp.sizeColumn(), 2);

        for (size_t i = 0; i < inp.sizeRow(); i++)
        {
            for (size_t j = 0; j < inp.sizeColumn(); j++)
            {
                data[i][j][0] = i;
                data[i][j][1] = j;
            }
        }
    }

    void resize(size_t row, size_t column, size_t tube)
    {
        data.resize(tube);
        for (int i = 0; i < tube; i++)
        {
            data[i] = Matrix<T>(row, column);
        }
    }

    Matrix<T>& operator[] (size_t tube)
    {
        return data[tube];
    }

    const Matrix<T>& operator[](size_t tube) const
    {
        return data[tube];
    }

    size_t depth()
    {
        return data.size();
    }

    size_t cols()
    {
        if (data.size() == 0)
            return 0;

        return data[0].sizeColumn();
    }

    size_t rows()
    {
        if (data.size() == 0 || data[0][0].size() == 0)
            return 0;

        return data[0][0].size();
    }

};
