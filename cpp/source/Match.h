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
#include <vector>

/// <summary>
/// A structure that describes the correspondence between two key points
/// </summary>
struct Match
{
    int src;
    int dst;
    float distance;
    int imgIdx;

    Match()
    {

    }

    Match(int n_src, int n_dst, double dist)
    {
        src = n_src;
        dst = n_dst;
        distance = dist;
    }

    Match(int n_src, int n_dst, double dist, int n_imgIdx)
    {
        src = n_src;
        dst = n_dst;
        distance = dist;
        imgIdx = n_imgIdx;
    }
};

typedef std::vector<Match> Matches;