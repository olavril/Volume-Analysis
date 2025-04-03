#pragma once

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

#include "communication.hpp"

double compute_variance(const std::vector<double> data, const double mean, const uint64_t NumParticles, const int Rank, const int NumRanks);

// function to compute a PDF from a data vector
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> computePDF(
                                                        const std::vector<double> data, const double vmin, const double vmax, 
                                                        const int numBins, const uint64_t numParticles, const int numProcessors, 
                                                        const int Rank, const int numRanks)
{   
    assert(("in the PDF computations, vmin must be smaller than vmax", vmin < vmax));
    // bin width
    const double BinWidth =( vmax - vmin)/(double)(numBins);
    // number of local particle
    uint64_t localNumParticles = data.size();
    // number of particles per processor
    const uint64_t numParticlesPerProcessor = std::ceil((double)(localNumParticles)/(double)(numProcessors));

    // result containers

    // mean data value per bin 
    std::vector<std::vector<double>> MeanDataDist(numProcessors);
    // number of gridpoints per bin
    std::vector<std::vector<uint64_t>> NumDist(numProcessors);
    // number of gridpoints outside (vmin,vmax)
    // purely diagnostic
    std::vector<uint64_t> NumOutsideIntervalDist(numProcessors, 0);

    // resize the results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        // resize the result
        MeanDataDist[iProc].resize(numBins);
        NumDist[iProc].resize(numBins);
    }

    //reset the result vectors
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        #pragma omp parallel for 
        for (size_t iBin = 0; iBin < numBins; iBin++){
            MeanDataDist[iProc][iBin] = 0.0;
            NumDist[iProc][iBin]      = 0.0;
        }
    }

    bool ErrPrint = true;
    // one process per processor
    #pragma omp parallel for
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iParticle = iProc*numParticlesPerProcessor; (iParticle < (iProc+1)*numParticlesPerProcessor) && (iParticle < localNumParticles); iParticle++){
            if ((vmin < data[iParticle]) && (data[iParticle] < vmax)){
                int iBinPart = std::floor((data[iParticle]-vmin)/BinWidth);
                MeanDataDist[iProc][iBinPart] += data[iParticle];
                NumDist[iProc][iBinPart]++;
            } else {
                NumOutsideIntervalDist[iProc]++;
                if ((ErrPrint && (Rank == 0)) && (iProc == 0)){
                    std::cout << "  data: " << data[iParticle] << std::endl;
                    ErrPrint = false;
                }
            }
        }
    }

    // mean data value per bin 
    std::vector<double> MeanData(numBins, 0.0);
    // number of gridpoints per bin
    std::vector<double> NumPart(numBins, 0.0);
    // number of gridpoints per bin
    std::vector<double> fracPart(numBins, 0.0);
    // number of gridpoints outside (vmin,vmax)
    uint64_t NumOutsideInterval = 0;

    // add up the processors
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iBin = 0; iBin < numBins; iBin++){
            MeanData[iBin] += MeanDataDist[iProc][iBin];
            NumPart[iBin]  += static_cast<double>(NumDist[iProc][iBin]);
        }
        NumOutsideInterval += NumOutsideIntervalDist[iProc];
    }
    if (NumOutsideInterval > 0){
        std::cout << Rank << ": number of gridpoints outside (" << vmin << "," << vmax <<"): " << NumOutsideInterval << std::endl;
    }

    // collapse the ranks to rank 0
    MeanData = collapse(MeanData, Rank, numRanks);
    NumPart  = collapse(NumPart,  Rank, numRanks);
    // synchronize the ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // compute the total mean of the data
    double mean = 0.0;
    if (Rank == 0){
        for (int iBin = numBins-1; iBin >= 0; iBin -= 1){
            // std::cout << iBin << std::endl;
            mean += MeanData[iBin];
        }
        mean = mean / (double)(numParticles);
    }
    // synchronize the ranks
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *)&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // compute the variance
    double var = compute_variance(data, mean, numParticles, Rank, numRanks);

    // vector containing the statistics
    // [0]: mean
    // [1]: variance
    std::vector<double> Statistics(2, 0.0);
    Statistics[0] = mean;
    Statistics[1] = var;


    // normalize
    #pragma omp parallel for
    for (size_t iBin = 0; iBin < numBins; iBin++){
        if (NumPart[iBin] > 0){
            MeanData[iBin] = MeanData[iBin]/NumPart[iBin];
            fracPart[iBin]  = NumPart[iBin] / ((double)(numParticles) * BinWidth);
        } else {
            MeanData[iBin] = ((double)(iBin)+0.5)*BinWidth + vmin;
            fracPart[iBin]  = 0.0;
        }
    }

    // return PDF
    return std::make_tuple(MeanData,NumPart,fracPart,Statistics);
}





// function to compute the variance of a given data vector
double compute_variance(const std::vector<double> data, const double mean, const uint64_t NumParticles, const int Rank, const int NumRanks)
{
    // number of particles per partial sum
    const double NumParticlesPerPartialSum = 10000;
    // number of partial sums
    const int NumPartialSums = std::ceil((double)(data.size()) / NumParticlesPerPartialSum);

    // vector that holds the results of the partial sums
    std::vector<double> PartialSums(NumPartialSums, 0.0);
    // compute the partial sums
    #pragma omp parallel for
    for (size_t iPartial = 0; iPartial < NumPartialSums; iPartial++){
        for (size_t iParticle = iPartial*NumParticlesPerPartialSum; iParticle < (iPartial+1)*NumParticlesPerPartialSum; iParticle++){
            PartialSums[iPartial] += std::pow((data[iParticle] - mean), 2);
        }
    }

    // sum of the local data
    double LocalSum = 0.0;
    // sum up the partial sums
    for (size_t iPartial = 0; iPartial < NumPartialSums; iPartial++){
        LocalSum += PartialSums[iPartial];
    }

    // total Sum of the data across all ranks
    double TotalSum = 0.0;
    // container for sending / recieving sum data
    double DataSumSendRecv = 0.0;
    // broadcasting local sums
    for (size_t iSender = 0; iSender < NumRanks; iSender++){
        if (Rank == iSender){
            DataSumSendRecv = LocalSum;
        }
        MPI_Bcast((void *)&DataSumSendRecv, 1, MPI_DOUBLE, iSender, MPI_COMM_WORLD);
        TotalSum += DataSumSendRecv;
    }

    // compute the variance from the sum
    TotalSum = TotalSum / (double)(NumParticles);
    // return the result
    return TotalSum;
} // end of compute_variance() function


// function to save PDFs
void save_PDF(const std::vector<double> MeanData, const std::vector<double> NumPart, const std::vector<double> PDF, const std::vector<double> Statistics, 
              const std::string SimFile, const std::string GridFile, const double time, const std::string key, const int numOutput, 
              const bool NearestNeighbor, const bool volume_weighted)
{
    // Output filename 
    std::string OutputFilename;
    if (volume_weighted){
        OutputFilename = "volume-PDF-" + key + "-" + std::to_string(numOutput) + ".txt";
    } else {
        OutputFilename = "mass-PDF-" + key + "-" + std::to_string(numOutput) + ".txt";
    }

    // open the file
    std::ofstream outFile(OutputFilename);
    if (outFile.is_open())
    {   
        // header
        outFile << "# simfile:  " << SimFile << std::endl;
        outFile << "# gridfile: " << GridFile << std::endl;
        outFile << "# time: " << time << std::endl;
        outFile << "# mean: " << Statistics[0] << std::endl;
        outFile << "# Variance: " << Statistics[1] << std::endl;
        outFile << "# nearest neighbor mapping: " << NearestNeighbor << std::endl;
        outFile << "# mean   numPart   PDF(" << key << ")" << std::endl;
     
        // loop over the bins 
        for (size_t iBin = 0; iBin <MeanData.size(); iBin++){
            //mean data in each bin
            outFile << MeanData[iBin] << " ";
            // number of grid Points n each bin
            outFile << NumPart[iBin] << " ";
            // total number of ceonnections per bin
            outFile << PDF[iBin] << std::endl;
        }
        // close the file
        outFile.close();
    }

    return;
}