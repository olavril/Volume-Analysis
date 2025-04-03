#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>
#include <chrono>

#include "MPI_utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"
#include "vector_operations.hpp"
#include "simulation_utils.hpp"

#include "heffte.h"

#include "grid.hpp"
#include "PDF.hpp"
#include "power_spectrum.hpp"
#include "structure_function.hpp"


void printHelp(char* binName, int rank);

using namespace sphexa;

int main(int argc, char** argv){
    // start the total runtime measurement
    auto TotalRuntimeStart = std::chrono::high_resolution_clock::now();
    // start the MPI environment
    auto [Rank,numRanks] = initMPI();
    

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // get arguments that were passed to the program --------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // get the settings
    const ArgParser parser(argc, (const char**)argv);
    // print help if -h was set
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], Rank);
        return MPIfinal();
    }

    const std::string DefaultSimFile  = "";
    const std::string DefaultGridFile = "";

    // path to the file conatining the simulation data ( required if no grid file is available)
    const std::string SimFile      = parser.get("--simfile", DefaultSimFile);
    // step number in the simulation file that should be used
    const int StepNo               = parser.get("--stepNo", 0);
    // path to the grid file containing the grid data 
    // if a SimFile was provided, the grid will be coputed and stored her
    // if no SimFile was proided, the grid data will be read from this file
    const std::string GridFile      = parser.get("--gridfile", DefaultGridFile);
    // number of grid points in 1 dimension 
    const int numGridPoints     = parser.get("--numGridPoints", 5000);
    // total number of grid points
    const uint64_t totalNumGridPoints = std::pow(numGridPoints,3);
    // number of processors per rank
    const int numTestPoints     = parser.get("--numTest", 10000);
    // number of processors per rank
    const int numProcessors     = parser.get("--numProcs", 1);
    // number that is used to identify tho output
    const int numOutput         = parser.get("--numOut", 0);
    // use the nearest neighbor approach for mapping the SPH-data to the grid?
    // false : SPH-interpolation (default)
    // true  : nearest-neighbor mapping
    bool NearestNeighbor = false;
    if (parser.exists("-nn") || parser.exists("--nn") || parser.exists("-NN") || parser.exists("--NN")){
        NearestNeighbor = true;
    }


    // check the inputs
    assert(("you must provide a --gridfile to store / load the grid data", GridFile != DefaultGridFile));
    assert(("the number of grid-points (--numGridPoints) must be larger than 0", numGridPoints > 0));
    assert(("the number of processors must be larger than 0", numProcessors > 0));


    std::chrono::duration<double> LoadingRuntime = TotalRuntimeStart - TotalRuntimeStart;
    std::chrono::duration<double> MassPDFRuntime = TotalRuntimeStart - TotalRuntimeStart;
    std::chrono::duration<double> GridMappingRuntime = TotalRuntimeStart - TotalRuntimeStart;
    std::chrono::duration<double> VolumePDFRuntime = TotalRuntimeStart - TotalRuntimeStart;
    std::chrono::duration<double> PowerSpectraRuntime = TotalRuntimeStart - TotalRuntimeStart;
    std::chrono::duration<double> StructureFunctionRuntime = TotalRuntimeStart - TotalRuntimeStart;

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // run information print --------------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (Rank == 0){
        std::cout << "SimFile:   " << SimFile << "\tStepNo: " << StepNo << std::endl;
        std::cout << "GridFile:  " << GridFile << std::endl;
        std::cout << "number of grid-points:  " << numGridPoints << std::endl;
        std::cout << "nearest neighbor mapping:  " << NearestNeighbor << std::endl;
        std::cout << "NumOutput: " << numOutput << std::endl;
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // load simulation from checkpoint ----------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // start loading the data
    auto LoadingRuntimeStart = std::chrono::high_resolution_clock::now();

    // total number of SPH-particles in the simulation
    uint64_t NumParticles;
    // local number of SPH-particles stored on this rank
    uint64_t localNumParticles;
    // physical time of the snapshot
    double time;
    // kernel Choice
    int kernelChoice;
    // x-positions of the SPH-particles
    std::vector<double> xSim;
    // y-positions of the SPH-particles
    std::vector<double> ySim;
    // z-positions of the SPH-particles
    std::vector<double> zSim;
    // x-velocities of the SPH-particles
    std::vector<double> vxSim;
    // y-velocities of the SPH-particles
    std::vector<double> vySim;
    // z-velocities of the SPH-particles
    std::vector<double> vzSim;
    // total velocities of the SPH-particles
    std::vector<double> vSim;
    // smoothing-lengths of the SPH-particles
    std::vector<double> hSim;
    // densities of the SPH-particles
    std::vector<double> rhoSim;
    // neighbor-count of the SPH-particles
    std::vector<double> ncSim;
    // total mass of the simulation
    double SimMass = 0.0;
    // total kinetic energy of the simulation
    double SimEkin = 0.0;

    // HDF5 reader
    auto reader = makeH5PartReader(MPI_COMM_WORLD);

    if (SimFile != DefaultSimFile){
        // open the simulation data
        if(Rank==0){std::cout << "reading the simulation data" << std::endl;}
        reader->setStep(SimFile, StepNo, FileMode::collective);
        // get particle numbers
        NumParticles = reader->globalNumParticles();
        localNumParticles = reader->localNumParticles();
        reader->stepAttribute("time", &time, 1);
        reader->stepAttribute("kernelChoice", &kernelChoice, 1);
        // resize the data vectors
        xSim.resize(localNumParticles);
        ySim.resize(localNumParticles);
        zSim.resize(localNumParticles);
        vxSim.resize(localNumParticles);
        vySim.resize(localNumParticles);
        vzSim.resize(localNumParticles);
        vSim.resize(localNumParticles);
        hSim.resize(localNumParticles);
        rhoSim.resize(localNumParticles);
        // ncSim.resize(localNumParticles);
        // load the data
        reader->readField("x",    xSim.data());
        reader->readField("y",    ySim.data());
        reader->readField("z",    zSim.data());
        reader->readField("vx",  vxSim.data());
        reader->readField("vy",  vySim.data());
        reader->readField("vz",  vzSim.data());
        reader->readField("h",    hSim.data());
        reader->readField("rho",rhoSim.data());
        // reader->readField("nc",  ncSim.data());
        // compute total velocities
        vSim = vector_pythagoras(vxSim, vySim, vzSim);
        // close the step 
        reader->closeStep();
        if (Rank == 0){std::cout << "kernelChoice: " << kernelChoice << "\n" << std::endl;}
        MPI_Barrier(MPI_COMM_WORLD);

        // compute properties of the simulation
        if(Rank==0){std::cout << "computing simulation properties" << std::endl;}
        auto SimProperties = computeSimProperties(vSim, NumParticles,numProcessors, Rank, numRanks);
        SimMass = std::get<0>(SimProperties);
        SimEkin = std::get<1>(SimProperties);
    }

    // stop loading the data
    auto LoadingRuntimeStop = std::chrono::high_resolution_clock::now();
    LoadingRuntime = LoadingRuntimeStop - LoadingRuntimeStart;

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // mass weighted statistics -----------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // mean smoothing length
    double hMean;
    if (SimFile != DefaultSimFile){
        // start computing mass-weighted PDFs
        auto MassPDFRuntimeStart = std::chrono::high_resolution_clock::now();

        // compute the PDFs
        if(Rank==0){std::cout << "computing mass-weighted h PDF" << std::endl;}
        auto [hMassMean,hMassNum,hMassPDF,hMassStat] =computePDF(hSim, 0,0.1,1000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing mass-weighted vx PDF" << std::endl;}
        auto [vxMassMean,vxMassNum,vxMassPDF,vxMassStat] =computePDF(vxSim, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing mass-weighted vy PDF" << std::endl;}
        auto [vyMassMean,vyMassNum,vyMassPDF,vyMassStat] =computePDF(vySim, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing mass-weighted vz PDF" << std::endl;}
        auto [vzMassMean,vzMassNum,vzMassPDF,vzMassStat] =computePDF(vzSim, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing mass-weighted v PDF" << std::endl;}
        auto [vMassMean,vMassNum,vMassPDF,vMassStat] =computePDF(vSim, 0,30,30000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing mass-weighted rho PDF" << std::endl;}
        auto [rhoMassMean,rhoMassNum,rhoMassPDF,rhoMassStat] =computePDF(rhoSim, 0,400,400000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        // if(Rank==0){std::cout << "computing mass-weighted nc PDF" << std::endl;}
        // auto [ncMassMean,ncMassNum,ncMassPDF,ncMassStat] =computePDF(ncSim, 0,200,200, NumParticles,numProcessors, Rank,numRanks);
        // MPI_Barrier(MPI_COMM_WORLD);

        // save the PDFs
        if (Rank == 0){
            save_PDF(hMassMean,hMassNum,hMassPDF,hMassStat,SimFile,GridFile,time,"h",numOutput,false,false);
            save_PDF(vxMassMean,vxMassNum,vxMassPDF,vxMassStat,SimFile,GridFile,time,"vx",numOutput,false,false);
            save_PDF(vyMassMean,vyMassNum,vyMassPDF,vyMassStat,SimFile,GridFile,time,"vy",numOutput,false,false);
            save_PDF(vzMassMean,vzMassNum,vzMassPDF,vzMassStat,SimFile,GridFile,time,"vz",numOutput,false,false);
            save_PDF(vMassMean,vMassNum,vMassPDF,vMassStat,SimFile,GridFile,time,"v",numOutput,false,false);
            save_PDF(rhoMassMean,rhoMassNum,rhoMassPDF,rhoMassStat,SimFile,GridFile,time,"rho",numOutput,false,false);
            // save_PDF(ncMassMean,ncMassNum,ncMassPDF,ncMassStat,SimFile,GridFile,time,"nc",numOutput,false,false);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        hMean = hMassStat[0];
        vSim.clear();
        vSim.shrink_to_fit();
        MPI_Barrier(MPI_COMM_WORLD);
        
        // remove ncSim data
        ncSim.clear();
        ncSim.shrink_to_fit();

        // stop computing mass-weighted PDFs
        auto MassPDFRuntimeStop = std::chrono::high_resolution_clock::now();
        MassPDFRuntime = MassPDFRuntimeStop - MassPDFRuntimeStart;
    }


    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // map the simulation to the grid -----------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // grid velocity in x-direction
    std::vector<double> vxGrid;
    // grid velocity in x-direction
    std::vector<double> vyGrid;
    // grid velocity in x-direction
    std::vector<double> vzGrid;
    // total grid velocity
    std::vector<double> vGrid;
    // grid velocity in x-direction
    std::vector<double> rhoGrid;
    // number of neighbors in x-direction
    std::vector<double> ncGrid;

    // map simulation data to grid in a simfile was provided
    if (SimFile != DefaultSimFile){
        // start of the grid mapping
        auto GridMappingRuntimeStart = std::chrono::high_resolution_clock::now();
        
        // map the simulation data to grid
        auto Grid = MapToGrid(xSim,ySim,zSim,vxSim,vySim,vzSim,hSim,rhoSim,numGridPoints,NumParticles,kernelChoice,numProcessors,Rank,numRanks, NearestNeighbor);
        if(Rank==0){std::cout << "Map to Grid Done" << std::endl;}
        MPI_Barrier(MPI_COMM_WORLD);
        vxGrid  = std::get<0>(Grid);
        vyGrid  = std::get<1>(Grid);
        vzGrid  = std::get<2>(Grid);
        rhoGrid = std::get<3>(Grid);
        ncGrid  = std::get<4>(Grid);
        if(Rank==0){std::cout << "Grid data unpacked" << std::endl;}
        vGrid = vector_pythagoras(vxGrid, vyGrid, vzGrid);
        if(Rank==0){std::cout << "total Grid velocities computed" << std::endl;}
        MPI_Barrier(MPI_COMM_WORLD);


        // save the grid data to the gridfile
        SaveGrid(vxGrid,vyGrid,vzGrid,rhoGrid,ncGrid,numGridPoints,time,NearestNeighbor,SimFile,GridFile, Rank,numRanks);
        
        // start of the grid mapping
        auto GridMappingRuntimeStop = std::chrono::high_resolution_clock::now();
        GridMappingRuntime = GridMappingRuntimeStop - GridMappingRuntimeStart;
    // if not, load the data from the grid file
    } 
    // TODO: loading


    // get properties of the grid
    auto [GridMass, GridEkin] = computeGridProperties(vGrid, rhoGrid, numGridPoints, numProcessors, Rank, numRanks);
    if (Rank == 0){
        std::cout << "\ninformation on grid mapping:" << std::endl;
        std::cout << "\t                 grid  /  SPH" << std::endl;
        std::cout << "\tmass:           " << GridMass << " / 1.0" << std::endl;
        std::cout << "\tkinetic energy: " << GridEkin << " / " << SimEkin << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (SimFile != DefaultSimFile){
        // clear unnecessary memory
        if(Rank==0){std::cout << "\nremoving SPH memory\n" << std::endl;}
        xSim.clear();
        ySim.clear();
        zSim.clear();
        vxSim.clear();
        vySim.clear();
        vzSim.clear();
        hSim.clear();
        rhoSim.clear();
        xSim.shrink_to_fit();
        ySim.shrink_to_fit();
        zSim.shrink_to_fit();
        vxSim.shrink_to_fit();
        vySim.shrink_to_fit();
        vzSim.shrink_to_fit();
        hSim.shrink_to_fit();
        rhoSim.shrink_to_fit();
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // volume weighted statistics ---------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (true){
        // stop computing volume-weighted PDFs
        auto VolumePDFRuntimeStart = std::chrono::high_resolution_clock::now();

        if(Rank==0){std::cout << "computing volume-weighted vx PDF" << std::endl;}
        auto [vxMean,vxNum,vxPDF,vxStat] =computePDF(vxGrid, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing volume-weighted vy PDF" << std::endl;}
        auto [vyMean,vyNum,vyPDF,vyStat] =computePDF(vyGrid, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing volume-weighted vz PDF" << std::endl;}
        auto [vzMean,vzNum,vzPDF,vzStat] =computePDF(vzGrid, -20,20,4000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing volume-weighted v PDF" << std::endl;}
        auto [vMean,vNum,vPDF,vStat] =computePDF(vGrid, 0,30,30000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing volume-weighted rho PDF" << std::endl;}
        auto [rhoMean,rhoNum,rhoPDF,rhoStat] =computePDF(rhoGrid, 0,400,400000, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing volume-weighted nc PDF" << std::endl;}
        double ncmin = 0.5;
        double ncmax = 200.5;
        int ncnum = 200;
        if (NearestNeighbor){
            ncmin = 0.0;
            ncmax = 0.05;
            ncnum = 1000;
        }
        auto [ncMean,ncNum,ncPDF,ncStat] =computePDF(ncGrid, ncmin,ncmax,ncnum, NumParticles,numProcessors, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);

        // save the PDFs
        if (Rank == 0){
            save_PDF(vxMean,vxNum,vxPDF,vxStat,SimFile,GridFile,time,"vx",numOutput,NearestNeighbor,true);
            save_PDF(vyMean,vyNum,vyPDF,vyStat,SimFile,GridFile,time,"vy",numOutput,NearestNeighbor,true);
            save_PDF(vzMean,vzNum,vzPDF,vzStat,SimFile,GridFile,time,"vz",numOutput,NearestNeighbor,true);
            save_PDF(vMean,vNum,vPDF,vStat,SimFile,GridFile,time,"v",numOutput,NearestNeighbor,true);
            save_PDF(rhoMean,rhoNum,rhoPDF,rhoStat,SimFile,GridFile,time,"rho",numOutput,NearestNeighbor,true);
            if (NearestNeighbor){
                save_PDF(ncMean,ncNum,ncPDF,ncStat,SimFile,GridFile,time,"nndist",numOutput,NearestNeighbor,true);
            } else {
                save_PDF(ncMean,ncNum,ncPDF,ncStat,SimFile,GridFile,time,"nc",numOutput,NearestNeighbor,true);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // stop computing volume-weighted PDFs
        auto VolumePDFRuntimeStop = std::chrono::high_resolution_clock::now();
        VolumePDFRuntime = VolumePDFRuntimeStop - VolumePDFRuntimeStart;
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // compute the power spectrum ---------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    if (true){
        // start computing power spectra
        auto PowerSpectraRuntimeStart = std::chrono::high_resolution_clock::now();

        if(Rank==0){std::cout << "computing power spectrum frequencies k" << std::endl;}  
        std::vector<double> k1D = computeK(numGridPoints);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing rho power spectrum" << std::endl;}  
        std::vector<double> rhoPS = computePS(rhoGrid, numGridPoints, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing vx power spectrum" << std::endl;}  
        std::vector<double> vxPS = computePS(vxGrid, numGridPoints, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing vy power spectrum" << std::endl;}  
        std::vector<double> vyPS = computePS(vyGrid, numGridPoints, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing vz power spectrum" << std::endl;}  
        std::vector<double> vzPS = computePS(vzGrid, numGridPoints, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        if(Rank==0){std::cout << "computing v² power spectrum" << std::endl;}  
        std::vector<double> vPS = computePS(vector_product(vector_square(vGrid), rhoGrid), numGridPoints, Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);

        // bin the power spectra
        const int numBins = numGridPoints/2;
        if(Rank==0){std::cout << "\nbinning rho power spectrum" << std::endl;}  
        auto [krho,PSrho,Numrho] = PSbinning(k1D,rhoPS,numGridPoints,numBins,Rank,numRanks);
        if(Rank==0){std::cout << "binning vx power spectrum" << std::endl;}  
        auto [kvx,PSvx,Numvx]    = PSbinning(k1D,vxPS,numGridPoints,numBins,Rank,numRanks);
        if(Rank==0){std::cout << "binning vy power spectrum" << std::endl;}  
        auto [kvy,PSvy,Numvy]    = PSbinning(k1D,vyPS,numGridPoints,numBins,Rank,numRanks);
        if(Rank==0){std::cout << "binning vz power spectrum" << std::endl;}  
        auto [kvz,PSvz,Numvz]    = PSbinning(k1D,vzPS,numGridPoints,numBins,Rank,numRanks);
        if(Rank==0){std::cout << "binning vx² + vy² + vz² power spectrum" << std::endl;}  
        auto [kv2,PSv2,Numv2]    = PSbinning(k1D,vector_sum(vector_square(vxPS),vector_square(vyPS),vector_square(vzPS)),numGridPoints,numBins,Rank,numRanks);
        if(Rank==0){std::cout << "binning v² power spectrum" << std::endl;}  
        auto [kv,PSv,Numv]    = PSbinning(k1D,vPS,numGridPoints,numBins,Rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);

        // save the power spectra
        if (Rank == 0){
            std::cout << "\nsaving power spectra to file" << std::endl;
            save_PS(krho,PSrho,Numrho, SimFile,GridFile, SimEkin,GridEkin, time,"rho",numOutput,NearestNeighbor);
            save_PS(kvx, PSvx, Numvx,  SimFile,GridFile, SimEkin,GridEkin, time, "vx",numOutput,NearestNeighbor);
            save_PS(kvy, PSvy, Numvy,  SimFile,GridFile, SimEkin,GridEkin, time, "vy",numOutput,NearestNeighbor);
            save_PS(kvz, PSvz, Numvz,  SimFile,GridFile, SimEkin,GridEkin, time, "vz",numOutput,NearestNeighbor);
            save_PS(kv2, PSv2, Numv2,  SimFile,GridFile, SimEkin,GridEkin, time, "v2",numOutput,NearestNeighbor);
            save_PS(kv,  PSv,  Numv,   SimFile,GridFile, SimEkin,GridEkin, time,  "v",numOutput,NearestNeighbor);
            std::cout << "power spectra saved\n" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // stop computing power spectra
        auto PowerSpectraRuntimeStop = std::chrono::high_resolution_clock::now();
        PowerSpectraRuntime = PowerSpectraRuntimeStop - PowerSpectraRuntimeStart;
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // compute the structure function -----------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    std::vector<double> SFdist;
    std::vector<double> SFval;
    std::vector<double> SFconn;

    if (true){
        // start computing structure functions
        auto StructureFunctionRuntimeStart = std::chrono::high_resolution_clock::now();

        if (Rank==0){std::cout << "\ncomputing velocity SF with hMean = " << hMean << std::endl;}
        if (Rank==0){std::cout << "number of test particles: " << numTestPoints << std::endl;}
        auto SF = computeSF(vxGrid,vyGrid,vzGrid, numGridPoints, numGridPoints/2,2*hMean, numTestPoints, numProcessors,Rank,numRanks);
        SFdist  = std::get<0>(SF);
        SFval   = std::get<1>(SF);
        SFconn  = std::get<2>(SF);
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank==0){std::cout << Rank << ": SF computation done" << std::endl;}
    
        if (Rank == 0){
            save_SF(SFdist,SFval,SFconn,numTestPoints,hMean, SimFile,GridFile,time, numOutput,NearestNeighbor,true);
        }

        // stop computing structure functions
        auto StructureFunctionRuntimeStop = std::chrono::high_resolution_clock::now();
        StructureFunctionRuntime = StructureFunctionRuntimeStop - StructureFunctionRuntimeStart;
    }


    
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // information on grid mapping --------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (Rank == 0){
        std::cout << "\ninformation on grid mapping:" << std::endl;
        std::cout << "\t                 grid  /  SPH" << std::endl;
        std::cout << "\tmass:           " << GridMass << " / 1.0" << std::endl;
        std::cout << "\tkinetic energy: " << GridEkin << " / " << SimEkin << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // runtime measurements ---------------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // stop the total runtime measurement
    auto TotalRuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> TotalRuntime = TotalRuntimeStop - TotalRuntimeStart;

    // print runtime information
    if ( Rank == 0){
        std::cout << "\ntotal runtime: " << TotalRuntime.count() << "sec" << std::endl;
        std::cout << "\tloading runtime:  " << LoadingRuntime.count() << "sec" << std::endl;
        if (MassPDFRuntime.count() > 0){         std::cout << "\tmass-PDF runtime:            " << MassPDFRuntime.count() << "sec" << std::endl;}
        if (GridMappingRuntime.count() > 0){     std::cout << "\tgrid-mapping runtime:        " << GridMappingRuntime.count() << "sec" << std::endl;}
        if (VolumePDFRuntime.count() > 0){       std::cout << "\tvolume-PDF runtime:          " << VolumePDFRuntime.count() << "sec" << std::endl;}
        if (PowerSpectraRuntime.count() > 0){    std::cout << "\tpower spectra runtime:       " << PowerSpectraRuntime.count() << "sec" << std::endl;}
        if (StructureFunctionRuntime.count() > 0){std::cout << "\tstructure function runtime: " << StructureFunctionRuntime.count() << "sec" << std::endl;}
    }


    // finalize the MPI environment
    return MPIfinal();
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--simfile \t\t HDF5 checkpoint file with simulation data. Needed for computng the grid if none is available\n\n");
        printf("\t--stepNo \t\t step number of the data in the SimFile. Default is 0.\n\n");
        printf("\t--gridfile \t\t HDF5 file containing the girid data. If a simfile is provided, the computed grid will be stored here.\n\n");
        printf("\t--numGridPoints \t\t number of grid points in 1 dimension. Default is 5000.\n\n");
        printf("\t--numProcs \t\t number of processors available to each rank. Default is 1.\n\n");
        printf("\t--numOut \t\t number that is used to identify the outputs from this run. Default is 0.\n\n");
    }
}