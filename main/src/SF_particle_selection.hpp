#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <cassert>

// function that returns the probability with which the particle should be selected
// input:   distance between the test particle and the potential connection particle
//          minimum distance required to be eligable for selection
// output:  probability for this particle to be selected (value between 0 and 1)
double selection_probability(const double dist, const double minDist){
    // maximum distance for selection
    const double maxDist = 0.5;
    // output value
    double probability;
    if ((dist < minDist) || (dist > maxDist)){
        probability = 0.0;
    } else {
        // normalisation value
        double norm = std::pow(minDist,2);
        probability = norm / std::pow(dist,2);
    }
    return probability;
}

// integrate the selection function above from minDist up to 0.5
// input:   minimum distance that should be used for the selection
//          number of processors available for the local rank
// output:  integral over the selection function
double integrated_selection_probability(const double minDist, const int numProcessors){
    // maximum distance for integration
    const double maxDist = 0.5;
    // number of function evaluations for the integration
    const unsigned int numEvaluations = 1000000;
    // distance step size
    const double distStep = (maxDist - minDist)/numEvaluations;
    // vector for sums of the function values (one entry per processor)
    std::vector<double> sums(numProcessors, 0.0);
    // loop over the distance interval and evaluate the selection function
    #pragma omp parallel for
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc+1; i < numEvaluations; i += numProcessors){
            double dist = maxDist - i*distStep;
            sums[iProc] += std::pow(dist,2) * selection_probability(dist,minDist);
        }
    }
    // total sum of function values
    double sum = 0.5 * (std::pow(minDist,2)*selection_probability(minDist,minDist) + std::pow(maxDist,2)*selection_probability(maxDist,minDist));
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        sum += sums[iProc];
    }
    sum = sum * 4*M_PI;
    return sum*distStep;
}

// function to decide if a given particle should be selected or not
// input:   distance of the parzicle to the test particle
//          minimum distance at which particles can be selected
//          normalisation value (will be multiplied onto the random number -> increases / decreases selection chance)
//          std::mt19937 Mersenne Twister for random generation
//          std::uniform_real_distribution<> for random generation
// output:  boolean value (true for selected, false for not selected)
bool selection(const double dist, const double minDist, const double norm){
    // should the particle be selected?
    bool selected;
    // selection probability
    const double prob = selection_probability(dist,minDist);
    if (prob > 0){
        // random seed
        std::random_device rd;
        // Mersenne Twister
        std::mt19937 gen(rd());
        // uniform distribution
        std::uniform_real_distribution<> dis(0.0,1.0);
        // generate a random value between 0 and 1
        const double rnd = dis(gen)/norm;
        if (rnd < prob){
            selected = true;
            //std::cout << prob << " > " << rnd << std::endl;
        } else {
            selected = false;
        }
    } else {
        selected = false;
    }
    return selected;
}


// uniform selection of test particles on the local rank
std::vector<uint64_t> selectTest(const uint64_t dataSize, const int numLocalTestParticles)
{
    assert(("when selecting test particles, you cannot select more particles than this node holds", dataSize > numLocalTestParticles));
    // create the result vector
    std::vector<uint64_t> result(numLocalTestParticles, 0);
    // number of already selected test particles
    int numSelectedParticles = 0;

    while (numSelectedParticles < numLocalTestParticles){
        // generate a random particle index
        // random seed
        std::random_device rd;
        // Mersenne Twister
        std::mt19937 gen(rd());
        // uniform distribution
        std::uniform_int_distribution<> dis(0,dataSize-1);
        uint64_t index = dis(gen);

        // check if the new index was already selected
        bool already_selected = false;
        for (size_t i = 0; i < numSelectedParticles; i++){
            if (index == result[i]){
                already_selected = true;
                break;
            }
        }

        // if it was not already selected, select it now
        if (not already_selected){
            result[numSelectedParticles] = static_cast<uint64_t>(index);
            numSelectedParticles++;
        }
    }

    return result;
}