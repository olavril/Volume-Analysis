#pragma once

#include <vector>
#include <cmath>
#include <stdint.h>
#include <omp.h>
#include <mpi.h>

// function to collapse a given data vector to rank 0
std::vector<double> collapse(std::vector<double> data, const int Rank, const int numRanks){
    // number of datapoints that need to be communicated
    const uint64_t numDataPoints = data.size();
    // create vector for recieving data
    std::vector<double> RecvData(numDataPoints);
    // number of ranks that still contain data (not collapsed)
    int numRanksRemaining = numRanks;
    // number of ranks that will remain after this collapse
    int nextNumRanksRemaining;

    // repeat the collapse as long as there are more than 1 remaining ranks
    while (numRanksRemaining > 1){
        // compute how many ranks will remain after this collapse
        nextNumRanksRemaining = std::ceil((double)(numRanksRemaining)/2);
        // synchronize the ranks
        MPI_Barrier(MPI_COMM_WORLD);
        // this rank recieves data
        if (Rank < nextNumRanksRemaining) {
            // rank from which this rank recieves data
            int RecvRank = Rank + nextNumRanksRemaining;
            // check if this rank will recieve data or not
            if (RecvRank < numRanksRemaining){
                // recieve the data
                MPI_Recv((void *)&RecvData[0], numDataPoints, MPI_DOUBLE, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                // add up the recieved data
                #pragma omp parallel for
                for (size_t i = 0; i < numDataPoints; i++){
                    data[i] += RecvData[i];
                    RecvData[i] = 0.0;
                }
            }
        // this rank sends its data
        } else if (Rank < numRanksRemaining){
            // rank to which this rank sends its data
            int SendRank = Rank - nextNumRanksRemaining;
            // send the data
            MPI_Send((void *)&data[0], numDataPoints, MPI_DOUBLE, SendRank, 1, MPI_COMM_WORLD);
        }
        // set the new number of remaining ranks
        numRanksRemaining = nextNumRanksRemaining;
    } // end of the collapse
    return data;
}

// function to collapse a given data value to rank 0
double collapse(double data, const int Rank, const int numRanks){
    // create vector for recieving data
    double RecvData;
    // number of ranks that still contain data (not collapsed)
    int numRanksRemaining = numRanks;
    // number of ranks that will remain after this collapse
    int nextNumRanksRemaining;

    // repeat the collapse as long as there are more than 1 remaining ranks
    while (numRanksRemaining > 1){
        // compute how many ranks will remain after this collapse
        nextNumRanksRemaining = std::ceil((double)(numRanksRemaining)/2);
        // synchronize the ranks
        MPI_Barrier(MPI_COMM_WORLD);
        // this rank recieves data
        if (Rank < nextNumRanksRemaining) {
            // rank from which this rank recieves data
            int RecvRank = Rank + nextNumRanksRemaining;
            // check if this rank will recieve data or not
            if (RecvRank < numRanksRemaining){
                // recieve the data
                MPI_Recv((void *)&RecvData, 1, MPI_DOUBLE, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                // add up the recieved data
                data += RecvData;
            }
        // this rank sends its data
        } else if (Rank < numRanksRemaining){
            // rank to which this rank sends its data
            int SendRank = Rank - nextNumRanksRemaining;
            // send the data
            MPI_Send((void *)&data, 1, MPI_DOUBLE, SendRank, 1, MPI_COMM_WORLD);
        }
        // set the new number of remaining ranks
        numRanksRemaining = nextNumRanksRemaining;
    } // end of the collapse
    return data;
}


// combine a distributed (double) vector, such that each rank holds the entire vector
std::vector<double> combine(std::vector<double> data, const int Rank, const int numRanks)
{
    // communicate how many datapoints each rank has

    // number of datapoints for each rank
    std::vector<uint64_t> numData(numRanks, 0);
    // total sum of grid points
    uint64_t totalNumDataPoints = 0;
    // send / recieve buffer
    uint64_t Buffer;
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        if (iRank == Rank){Buffer = static_cast<uint64_t>(data.size());}
        MPI_Bcast((void *)&Buffer, 1, MPI_UNSIGNED_LONG, iRank, MPI_COMM_WORLD);
        numData[iRank]      = Buffer;
        totalNumDataPoints += Buffer;
    }

    // communicate the data

    // new data vector
    std::vector<double> newData(totalNumDataPoints);
    // number of assigned data points
    uint64_t numAssigned = 0;
    // send/recieve buffer for vectors
    std::vector<double> VecBuffer;
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        VecBuffer.resize(numData[iRank]);
        if (iRank == Rank){VecBuffer = data;}
        MPI_Bcast((void *)&VecBuffer[0], 1, MPI_DOUBLE, iRank, MPI_COMM_WORLD);
        for (size_t iPoint = 0; iPoint < numData[iRank]; iPoint++){
            newData[numAssigned] = VecBuffer[iPoint];
            numAssigned++;
        }
    }

    return newData;
}

// combine a distributed (int) vector, such that each rank holds the entire vector
std::vector<int> combine(std::vector<int> data, const int Rank, const int numRanks)
{
    // communicate how many datapoints each rank has

    // number of datapoints for each rank
    std::vector<uint64_t> numData(numRanks, 0);
    // total sum of grid points
    uint64_t totalNumDataPoints = 0;
    // send / recieve buffer
    uint64_t Buffer;
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        if (iRank == Rank){Buffer = static_cast<uint64_t>(data.size());}
        MPI_Bcast((void *)&Buffer, 1, MPI_UNSIGNED_LONG, iRank, MPI_COMM_WORLD);
        numData[iRank]      = Buffer;
        totalNumDataPoints += Buffer;
    }

    // communicate the data

    // new data vector
    std::vector<int> newData(totalNumDataPoints);
    // number of assigned data points
    uint64_t numAssigned = 0;
    // send/recieve buffer for vectors
    std::vector<int> VecBuffer;
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        VecBuffer.resize(numData[iRank]);
        if (Rank == iRank){
            for (size_t iPoint = 0; iPoint < data.size(); iPoint++){
                VecBuffer[iPoint] = data[iPoint];
            }
        }
        MPI_Bcast((void *)&VecBuffer[0], numData[iRank], MPI_INT, iRank, MPI_COMM_WORLD);
        for (size_t iPoint = 0; iPoint < VecBuffer.size(); iPoint++){
            newData[numAssigned] = VecBuffer[iPoint];
            numAssigned++;
        }
    }

    return newData;
}