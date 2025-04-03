#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <cassert>
#include <omp.h>

#include "communication.hpp"
#include "grid.hpp"
#include "SF_particle_selection.hpp"

std::tuple<int,int,int> select_start(const uint64_t iTest, const int num);


// function to compute the structure function for a given data vector
// input:   data for which the SF should be computed
//          number of grid points (in one dimension)
//          number of distance bins
//          minimum distance (particles hat are closer are discarded, set to ~resolution length)
//          number of processors available to each rank
//          local Rank
//          number of ranks
// output:  tuple containing the distances, SF values and the number of connections for eaach bin
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>> computeSF(
                const std::vector<double> xdata, const std::vector<double> ydata, const std::vector<double> zdata, 
                const int numGridPoints, const int numBins, const double minDist, const int numTestParticles,
                const int numProcessors, const int Rank, const int numRanks)
{   
    // box of the full domain
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    // find a processor grid that minimizes the total surface area of all boxes
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    // get all boxes by splitting the Indizes onto the Processor Grid
    // local Box stored at entry Rank
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
    heffte::box3d<> LocalBox   = AllBoxes[Rank];

    // number of test points per processor
    const uint64_t numTestPerProcessor = std::ceil(static_cast<double>(numTestParticles) / static_cast<double>(numProcessors));

    // size of the Box in 1 dimension
    const double BoxSize = 1.0;
    // step size of the grid in 1 dimension
    const double GridStep = BoxSize / static_cast<double>(numGridPoints+1);

    // Bin step size
    const double BinStep = 0.5 / static_cast<double>(numBins);

    // --------------------------------------------------------------------------------------------------------------------------------------------------------
    // select test particles ----------------------------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------------------------------------------------

    // local number of test particles
    int localNumTestParticles = std::floor((double)(numTestParticles) / (double)(numRanks));
    if (Rank == numRanks-1){localNumTestParticles = numTestParticles - (numRanks-1)*localNumTestParticles;}
    //std::cout << Rank << ": selecting " << localNumTestParticles << " test particles" << std::endl;
    // local test particle containers
    std::vector<int> ixTest(localNumTestParticles), iyTest(localNumTestParticles), izTest(localNumTestParticles);
    std::vector<double> xdataTest(localNumTestParticles), ydataTest(localNumTestParticles), zdataTest(localNumTestParticles);
    // select the local test particles
    std::vector<uint64_t> TestIndizes = selectTest(static_cast<uint64_t>(xdata.size()), localNumTestParticles);
    for (size_t i = 0; i < localNumTestParticles; i++){
        auto [ix, iy, iz] = decomposeGridIndex(TestIndizes[i], 
                                        LocalBox.low[0],LocalBox.high[0],
                                        LocalBox.low[1],LocalBox.high[1],
                                        LocalBox.low[2],LocalBox.high[2]);
        // if (ix > 200){std::cout << "Test ix=" << ix << " > 200" << std::endl;}
        // if (iy > 200){std::cout << "Test iy=" << iy << " > 200" << std::endl;}
        // if (iz > 200){std::cout << "Test iz=" << iz << " > 200" << std::endl;}
        ixTest[i] = ix;
        iyTest[i] = iy;
        izTest[i] = iz;
        xdataTest[i] = xdata[TestIndizes[i]];
        ydataTest[i] = ydata[TestIndizes[i]];
        zdataTest[i] = zdata[TestIndizes[i]];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // communicate the test particles to all ranks
    if (numRanks > 1){
        ixTest    = combine(ixTest,    Rank,numRanks);
        iyTest    = combine(iyTest,    Rank,numRanks);
        izTest    = combine(izTest,    Rank,numRanks);
        xdataTest = combine(xdataTest, Rank,numRanks);
        ydataTest = combine(ydataTest, Rank,numRanks);
        zdataTest = combine(zdataTest, Rank,numRanks);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout << Rank << ": test particles combined (numTestParticles=" << ixTest.size() << ")" << std::endl;

    // for (size_t iRank = 0; iRank < numRanks; iRank++){
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (Rank == iRank){
    //         std::cout << " ix of test particles on Rank " << Rank << ":" << std::endl;
    //         for (size_t i = 0; i < 20; i++){
    //             std::cout << " " << ixTest[i];
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);




    // compute the SF

    // distributed result containers
    std::vector<std::vector<double>> distDist(numProcessors);
    std::vector<std::vector<double>> SFDist(numProcessors);
    std::vector<std::vector<double>> numDist(numProcessors);
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        distDist[iProc].resize(numBins);
        SFDist[iProc].resize(numBins);
        numDist[iProc].resize(numBins);
    }
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iBin = 0; iBin < numBins; iBin++){
            distDist[iProc][iBin] = 0.0;
            SFDist[iProc][iBin]   = 0.0;
            numDist[iProc][iBin]  = 0.0;
        }
    }
    // final combined results
    std::vector<double> distances(numBins, 0.0);
    std::vector<double> SF(numBins, 0.0);
    std::vector<double> Conns(numBins, 0.0);


    // loop over all test particles
    std::cout << Rank << ": computing SF" << std::endl;
    int increment = 11;
    #pragma omp parallel for 
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iTest = iProc*numTestPerProcessor; (iTest < (iProc+1)*numTestPerProcessor) && (iTest < numTestParticles); iTest++){
            // std::cout << "   iTest = " << iTest << "/" << numTestParticles << " ix=" << ixTest[iTest] << " iy=" << iyTest[iTest] << " iz=" << izTest[iTest] << std::endl;
            auto [ixstart, iystart, izstart] = select_start(iTest, increment);
            // loop over the local pGridpoints
            for (size_t ix = LocalBox.low[0]+ixstart; ix <= LocalBox.high[0]; ix += increment){
                for (size_t iy = LocalBox.low[1]+iystart; iy <= LocalBox.high[1]; iy += increment){
                    for (size_t iz = LocalBox.low[2]+izstart; iz <= LocalBox.high[2]; iz += increment){
                        // if (iTest >= 5){std::cout << "      ix=" << ix << " iy=" << iy << " iz=" << iz << std::endl;}
                        // get the local Grid index
                        uint64_t localGridIndex = getGridIndex(ix, iy, iz,
                                            LocalBox.low[0],LocalBox.high[0],
                                            LocalBox.low[1],LocalBox.high[1],
                                            LocalBox.low[2],LocalBox.high[2]);
                        // if (iTest >= 5){std::cout << "      localGridIndex=" << localGridIndex << std::endl;}
                        // compute distance
                        double dx   = correct_periodic_cond(getGridPosition(ixTest[iTest], GridStep) - getGridPosition(ix, GridStep));
                        double dy   = correct_periodic_cond(getGridPosition(iyTest[iTest], GridStep) - getGridPosition(iy, GridStep));
                        double dz   = correct_periodic_cond(getGridPosition(izTest[iTest], GridStep) - getGridPosition(iz, GridStep));
                        double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                        // if (iTest >= 5){std::cout << "      dx=" << dx << " dy=" << dy << " dz=" << dz << " dist=" << dist << std::endl;}
                        // is the current connection selected?
                        bool selected = selection(dist, minDist, 1.0);
                        if (selected){
                            // find the Bin index
                            int iBin = std::floor(dist / BinStep);
                            // if (iTest >= 5){std::cout << "      iBin=" << iBin << std::endl;}
                            // compute the SF contribution from ths particle
                            distDist[iProc][iBin] += dist;
                            SFDist[iProc][iBin]   += std::pow((xdataTest[iTest] - xdata[localGridIndex]), 2) 
                                                     + std::pow((ydataTest[iTest] - ydata[localGridIndex]), 2)
                                                     + std::pow((zdataTest[iTest] - zdata[localGridIndex]), 2);
                            numDist[iProc][iBin]  += 1.0;
                        }
                    } // end of iz loop
                } // end of iy loop
            } // end of ix loop
        } // end of test point loop
    } // end of the processor loop

    // final results
    //std::cout << Rank << ": combining local SF results" << std::endl;
    // merge the individual processor results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iBin = 0; iBin < numBins; iBin++){
            distances[iBin] += distDist[iProc][iBin];
            SF[iBin]        += SFDist[iProc][iBin];
            Conns[iBin]     += numDist[iProc][iBin];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // collapse results onto rank 0
    if (numRanks > 1){
        if (Rank == 0){std::cout << "collapsing SF results" << std::endl;}
        distances = collapse(distances, Rank,numRanks);
        SF        = collapse(SF, Rank,numRanks);
        Conns     = collapse(Conns, Rank,numRanks);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // rank 0 computes the mean distance & SF for each bin
    if (Rank == 0){
        std::cout << "compute mean bin SF results" << std::endl;
        uint64_t numTotalConn = 0;
        for (size_t iBin = 0; iBin < numBins; iBin++){
            distances[iBin] = distances[iBin] / Conns[iBin];
            SF[iBin]        = SF[iBin]        / Conns[iBin];
            numTotalConn += Conns[iBin];
        }
        std::cout << "\nstatistics on the SF computation:" << std::endl;
        std::cout << "\tnumber of test particles:    " << numTestParticles << std::endl;
        std::cout << "\tnumber of total connections: " << numTotalConn << "\n" << std::endl;
    }

    // return the results
    return std::make_tuple(distances, SF, Conns);
}

// select start offset for a given test particle
// this is used to reduce the number of local particles that are looped over while maintaining good coverage
std::tuple<int,int,int> select_start(const uint64_t iTest, const int num)
{   
    uint64_t uint_num = static_cast<uint64_t>(num);
    uint64_t uint_num2 = static_cast<uint64_t>(num*num);
    // start offsets that will be selected
    int izstart = std::floor(iTest / uint_num2);
    int iystart = std::floor((iTest - izstart*uint_num2) / uint_num);
    int ixstart = static_cast<int>(iTest - izstart*uint_num2 - iystart*uint_num);
    izstart = izstart % num;

    return std::make_tuple(ixstart,iystart,izstart);
}

// function to save PDFs
void save_SF(const std::vector<double> distance, const std::vector<double> SF, const std::vector<double> Conn, const int numTestPoints, const double hmean,
            const std::string SimFile, const std::string GridFile, const double time, const int numOutput, 
            const bool NearestNeighbor, const bool volume_weighted)
{
    // Output filename 
    std::string OutputFilename;
    if (volume_weighted){
        OutputFilename = "volume-SF-" + std::to_string(numOutput) + ".txt";
    } else {
        OutputFilename = "mass-SF-" + std::to_string(numOutput) + ".txt";
    }

    // open the file
    std::ofstream outFile(OutputFilename);
    if (outFile.is_open())
    {   
        // header
        outFile << "# simfile:  " << SimFile << std::endl;
        outFile << "# gridfile: " << GridFile << std::endl;
        outFile << "# time: " << time << std::endl;
        outFile << "# numTestPoints: " << numTestPoints << std::endl;
        outFile << "# hmean: " << hmean << std::endl;
        outFile << "# nearest neighbor mapping: " << NearestNeighbor << std::endl;
        outFile << "# dist   SF   #Conn" << std::endl;
     
        // loop over the bins 
        for (size_t iBin = 0; iBin < distance.size(); iBin++){
            //mean data in each bin
            outFile << distance[iBin] << " ";
            // number of grid Points n each bin
            outFile << SF[iBin] << " ";
            // total number of ceonnections per bin
            outFile << Conn[iBin] << std::endl;
        }
        // close the file
        outFile.close();
    }

    return;
}