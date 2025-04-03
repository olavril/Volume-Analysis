#pragma once

#include <iostream>
#include <vector>
#include <stdint.h>
#include <cmath>
#include <omp.h>

#include "communication.hpp"

std::tuple<double,double> computeSimProperties(const std::vector<double> v, 
                const int numParticles, const int numProcessors, const int Rank, const int numRanks)
{   
    // local grid-points per PRocessor
    const uint64_t NumParticlesPerProc = std::ceil(static_cast<double>(v.size()) / static_cast<double>(numProcessors));
    // mass of a single SPH particle
    const double mass = 1/static_cast<double>(numParticles);


    // total mass of the simulation
    double SimMass = 1.0;


    // total kinetic energy of the grid
    double SimEkin = 0.0;
    std::vector<double> SimDist(numProcessors, 0.0);
    // sum up the local grid point densities
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*NumParticlesPerProc;(i < (iProc+1)*NumParticlesPerProc) && (i < v.size()); i++){
            SimDist[iProc] += std::pow(v[i],2);
        }
    }
    // sum up the local results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        SimEkin += SimDist[iProc];
        SimDist[iProc] = 0.0;
    }
    // collapse the results of the different ranks
    SimEkin = collapse(SimEkin, Rank,numRanks);
    // multiply the sum of densities with the volume per grid point
    SimEkin = 0.5 * mass * SimEkin;


    // return the results
    return std::make_tuple(SimMass,SimEkin);
}