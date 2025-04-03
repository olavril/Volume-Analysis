#pragma once

#include <vector>
#include <cmath>
#include <cassert>

// function to compute the (element-wise) square of a vector
std::vector<double> vector_square(const std::vector<double> data)
{
    std::vector<double> result(data.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data.size(); i++){
        result[i] = std::pow(data[i],2);
    }
    return result;
}

// function to compute the (element-wise) sum of two (double) vectors
std::vector<double> vector_sum(const std::vector<double> data1, const std::vector<double> data2)
{
    assert(("the vectors that should be summed up (element wise) need to have the same size", data1.size() == data2.size()));
    std::vector<double> result(data1.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data1.size(); i++){
        result[i] = data1[i] + data2[i];
    }
    return result;
}

// function to compute the (element-wise) sum of three (double) vectors
std::vector<double> vector_sum(const std::vector<double> data1, const std::vector<double> data2, const std::vector<double> data3)
{
    assert(("the vectors that should be summed up (element wise) need to have the same size (1/2 mismatch)", data1.size() == data2.size()));
    assert(("the vectors that should be summed up (element wise) need to have the same size (2/3 mismatch)", data2.size() == data3.size()));
    std::vector<double> result(data1.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data1.size(); i++){
        result[i] = data1[i] + data2[i] + data3[i];
    }
    return result;
}

// function to compute the (element-wise) product of two (double) vectors
std::vector<double> vector_product(const std::vector<double> data1, const std::vector<double> data2)
{
    assert(("the vectors that should be multiplied (element wise) need to have the same size", data1.size() == data2.size()));
    std::vector<double> result(data1.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data1.size(); i++){
        result[i] = data1[i] * data2[i];
    }
    return result;
}

// function to compute the (element-wise) product of a (double) vector and a double value
std::vector<double> vector_product(const std::vector<double> data, const double factor)
{
    std::vector<double> result(data.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data.size(); i++){
        result[i] = data[i] * factor;
    }
    return result;
}

// function to compute the radial length of vectors from x, y and z data
// used for combining 1D velocities to the total velocity, etc.
std::vector<double> vector_pythagoras(const std::vector<double> xdata, const std::vector<double> ydata, const std::vector<double> zdata)
{
    assert(("the x and y vectors that for pythagoras need to have the same size", xdata.size() == ydata.size()));
    assert(("the x and z vectors that for pythagoras need to have the same size", xdata.size() == zdata.size()));
    std::vector<double> result(xdata.size());
    #pragma omp parallel for
    for(size_t i = 0; i < result.size(); i++){
        result[i] = std::sqrt(std::pow(xdata[i],2) + std::pow(ydata[i],2) + std::pow(zdata[i],2));
    }
    return result;
}