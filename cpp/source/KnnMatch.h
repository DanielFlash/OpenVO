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
#include <algorithm>

#include "TypeVOext.h"

using namespace std;

class KnnMatch
{
private:
	/// <summary>
	/// Getting N best Matches by distance
	/// </summary>
	/// <param name="distances">Vector of distance pairs, match number</param>
	/// <param name="count">Number of pairs to get</param>
	/// <returns>vector<pair<double, int>></returns>
	vector<pair<double, int>> getBest(vector<pair<double, int>> distances, int count)
	{
		std::sort(distances.begin(), distances.end());
		std::vector<pair<double, int>> best;
		for (int i = 0; i < distances.size(); i++)
		{
			if (!(distances[i].first < 0))
				best.push_back(distances[i]);

			if (best.size() == count)
				break;
		}

		return best;
	}

	/// <summary>
	/// Calculate the Euclidean distance between two descriptors
	/// </summary>
	/// <param name="first">Descriptor of keypoint 1</param>
	/// <param name="second">Descriptor of keypoint 2</param>
	/// <returns>double. -1 if the number of elements in the descriptors does not match</returns>
	double calcEuclideanDistance(Description first, Description second)
	{
		if (first.numbers.size() != second.numbers.size())
			return -1;

		double euclideanDistance = 0;
		for (size_t i = 0; i < first.numbers.size(); i++)
		{
			euclideanDistance += pow(first.numbers[i] - second.numbers[i], 2);
		}

		euclideanDistance = sqrt(euclideanDistance);

		return euclideanDistance;
	}
public:
	/// <summary>
	/// Perform KnnMatch on two sets of descriptors
	/// </summary>
	/// <param name="src">Descriptor Vector 1</param>
	/// <param name="dst">Descriptor Vector 2</param>
	/// <param name="numberNN">Number of Knn</param>
	/// <returns>Matches</returns>
	Matches find(Descriptions src, Descriptions dst, int numberNN)
	{
		Matches matches;
		vector<pair<double, int>> distances;
		double d;

		for(int i = 0; i < src.size(); i++)
		{
			distances.clear();
			for (int j = 0; j < dst.size(); j++)
			{
				d = calcEuclideanDistance(src[i], dst[j]);
				distances.push_back(pair<double, int>(d, j));
			}

			vector<pair<double, int>> best = getBest(distances, numberNN);
			for (int k = 0; k < best.size(); k++)
			{
				matches.push_back(Match(i, best[k].second, best[k].first));
			}
		}

		return matches;
	}
};