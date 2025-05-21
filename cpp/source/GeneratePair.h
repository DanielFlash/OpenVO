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
#include <ctime>
#include <set>
#include <utility>
#include <random>
#include <unordered_set>

using namespace std;

/// <summary>
/// Random keypoint pair generator
/// Designed for RANSAC and LMEDS filters
/// </summary>
class UniqueCombinationGenerator 
{
private:
    int maxValue;
    int sizeComb;
    mt19937 rng;
    set<vector<int>> generatedCombinations;
    size_t totalCombinations = 0;

    vector<int> generateUniqueSortedValues(int count) 
    {
        unordered_set<int> used;
        vector<int> result;
        result.reserve(count);

        uniform_int_distribution<int> dist(0, maxValue);
        while (result.size() < count) 
        {
            int value = dist(rng);
            if (used.insert(value).second) 
            {
                result.push_back(value);
            }
        }

        sort(result.begin(), result.end());
        return result;
    }

    size_t combinationCount(int n, int k) 
    {
        size_t result = 1;
        for (int i = 1; i <= k; ++i) 
        {
            result *= (n - (k - i));
            result /= i;
        }

        return result;
    }

public:
    UniqueCombinationGenerator(size_t maxValue, size_t sizeCombination)
        : maxValue(maxValue - 1), sizeComb(sizeCombination), rng(std::random_device{}())
    {
        if (maxValue == 0 || sizeCombination == 0)
            throw "maxValue or sizeCombination can be zero";
        if (maxValue < sizeCombination)
            throw "Size generate combination exceeds maxValue";

        totalCombinations = combinationCount(maxValue, sizeCombination);
    }

    size_t countMaxCombination()
    {
        return totalCombinations;
    }

    vector<int> generate()
    {
        vector<int> combination;

        if (generatedCombinations.size() >= totalCombinations)
            return combination;
       
        do
        {
            combination = generateUniqueSortedValues(sizeComb);
        } while (!generatedCombinations.insert(combination).second);

        return combination;
    }

    void reset()
    {
        generatedCombinations.clear();
        totalCombinations = 0;
    }
};