#include <vector>
#include <iostream>
#include <assert.h>
#include <tbb/concurrent_queue.h>
#include <omp.h>

#include "Event.h"
#include "GPUHitsAndDoublets.h"
#include "GPUCACell.h"
#include "parser.h"
#include "cuda.h"
#include "kernels.h"
#include "host_kernels.h"
static void show_usage(std::string name)
{
    std::cerr << "\nUsage: " << name << " <option(s)>" << " Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-n <number of events>\tSpecify the number of events to process [default: as many there are in the file]\n"
            << "\t-i <inputFile>\tSpecify the path of the input ASCII file containing the events to process [default: ../input/parsed.out]\n"
            << std::endl;

}

int main(int argc, char** argv)
{

    if (argc < 2)
    {
        show_usage(argv[0]);
        return 1;
    }

    unsigned int maxEvents = 10;
    std::string inputFile = "../input/parsed_noPU_fix.txt";
    unsigned int numberOfCUDAStreams = 5;
    unsigned int numberOfEventsPerStreamPerIteration = 1;
    unsigned int numberOfIterations = 1;
    unsigned int numberOfCPUThreads = 1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            show_usage(argv[0]);
            return 0;
        }
        else if (arg == "-n")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> maxEvents))
                {
                    std::cerr << "Invalid number " << argv[i] << '\n';
                    exit(1);

                }
            }
        }
        else if (arg == "-i")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> inputFile))
                {
                    std::cerr << "Invalid string " << argv[i] << '\n';
                    exit(1);

                }
            }
        }

        else if (arg == "-s")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> numberOfCUDAStreams))
                {
                    std::cerr << "Invalid number " << argv[i] << '\n';
                    exit(1);

                }
            }
        }

        else if (arg == "-t")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> numberOfIterations))
                {
                    std::cerr << "Invalid number " << argv[i] << '\n';
                    exit(1);

                }
            }
        }

        else if (arg == "-j")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> numberOfCPUThreads))
                {
                    std::cerr << "Invalid number " << argv[i] << '\n';
                    exit(1);

                }
            }
        }

        else if (arg == "-b")
        {
            if (i + 1 < argc) // Make sure we aren't at the end of argv!
            {
                i++;
                std::istringstream ss(argv[i]);
                if (!(ss >> numberOfEventsPerStreamPerIteration))
                {
                    std::cerr << "Invalid number " << argv[i] << '\n';
                    exit(1);

                }
            }
        }

    }

    std::vector<HostEvent> hostEvents;
    Region* h_regionParams;
    cudaMallocHost(&h_regionParams, sizeof(Region));

    if (maxEvents > 0)
    {
        hostEvents.reserve(maxEvents);
    }

    constexpr unsigned int maxNumberOfQuadruplets = 3000;
    constexpr unsigned int maxCellsPerHit = 100;
    unsigned int maxNumberOfHits = 0;
    unsigned int maxNumberOfDoublets = 0;
    parseinputFile(inputFile, hostEvents, *h_regionParams, maxEvents, maxNumberOfHits,
            maxNumberOfDoublets);
    unsigned int nEvents = hostEvents.size();
    std::cout << "Correctly parsed file containing " << nEvents << " events." << std::endl;

//    std::cout << "max number of hits and doublets in file " << maxNumberOfHits << " "
//            << maxNumberOfDoublets << std::endl;
//
//    std::cout << "hostEvents contains " << hostEvents.size() << " events" <<  std::endl;
//
//    for(auto& ev: hostEvents)
//    {
//        std::cout << "event id " <<ev.eventId << " rootlayers " << ev.rootLayers.size() << " hitsLayers " << ev.hitsLayers.size() <<
//                " layerPairs " <<  ev.doublets.size() << std::endl;
//        for(auto& rl : ev.rootLayers)
//            std::cout<< "root layer: " << rl << std::endl;
//        for(auto& hl : ev.hitsLayers)
//        {
//            std::cout<< "hits on layer: " << hl.size << " " << hl.x.size() << std::endl;
//
//            for(int jj = 0; jj < hl.x.size(); ++jj)
//                std::cout<< "hit " << jj << hl.x[jj] << " " << hl.y[jj] << " " << hl.z[jj] << std::endl;
//
//
//        }
//        for(auto& dl : ev.doublets)
//            {
//                std::cout<< "doublets on layer pair: " << dl.innerLayerId << " " << dl.outerLayerId << " " << dl.size << " " << dl.indices.size() << std::endl;
//                    for(unsigned int i = 0; i<dl.size; ++i )
//                    {
//                        std::cout<< "\t hits in doublet: "<< i << " " <<  dl.indices[2*i] << " "<< dl.indices[2*i+1] <<" " << ev.hitsLayers[dl.innerLayerId].x[dl.indices[2*i]] << " "<< ev.hitsLayers[dl.innerLayerId].y[dl.indices[2*i]] << " "<<ev.hitsLayers[dl.innerLayerId].z[dl.indices[2*i]]<< " "
//                         << " \t\t" << ev.hitsLayers[dl.outerLayerId].x[dl.indices[2*i+1]] << " "<< ev.hitsLayers[dl.outerLayerId].y[dl.indices[2*i+1]] << " "<<ev.hitsLayers[dl.outerLayerId].z[dl.indices[2*i+1]]<< std::endl;
//                    }
//            }
//    }

    std::cout << "preallocating pinned memory on host" << std::endl;

    unsigned int eventsPreallocatedOnGPU = numberOfEventsPerStreamPerIteration
            * numberOfCUDAStreams;
    unsigned int maxNumberOfLayers = 10;
    unsigned int maxNumberOfLayerPairs = 13;
    unsigned int maxNumberOfRootLayerPairs = 3;

    constexpr const float theThetaCut = 0.002f;
    constexpr const float thePhiCut = 0.2f;
    constexpr const float theHardPtCut = 0.0f;

    // HOST ALLOCATIONS FOR THE INPUT
    //////////////////////////////////////
    GPUEvent *h_allEvents;
    unsigned int* h_indices;
    GPULayerDoublets* h_doublets;
    cudaMallocHost(&h_allEvents, nEvents * sizeof(GPUEvent));
    //per each event per each layerPair you have a max number of doublets
    gpuErrchk(
            cudaMallocHost(&h_indices,
                    nEvents * maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int)));
    cudaMallocHost(&h_doublets, nEvents * maxNumberOfLayerPairs * sizeof(GPULayerDoublets));

    //per each event per each layer you have a max number of hits x y z
    GPULayerHits* h_layers;
    float *h_x, *h_y, *h_z;
    unsigned int* h_rootLayerPairs;

    cudaMallocHost(&h_layers, nEvents * maxNumberOfLayers * sizeof(GPULayerHits));
    gpuErrchk(cudaMallocHost(&h_x, nEvents * maxNumberOfLayers * maxNumberOfHits * sizeof(float)));
    cudaMallocHost(&h_y, nEvents * maxNumberOfLayers * maxNumberOfHits * sizeof(float));
    cudaMallocHost(&h_z, nEvents * maxNumberOfLayers * maxNumberOfHits * sizeof(float));
    cudaMallocHost(&h_rootLayerPairs, nEvents * maxNumberOfRootLayerPairs * sizeof(int));

    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * h_foundNtuplets;
    cudaMallocHost(&h_foundNtuplets,
            eventsPreallocatedOnGPU * sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));

    for (unsigned int i = 0; i < nEvents; ++i)
    {
        h_allEvents[i].eventId = hostEvents[i].eventId;

        h_allEvents[i].numberOfRootLayerPairs = 0;

        h_allEvents[i].numberOfLayers = hostEvents[i].hitsLayers.size();
        h_allEvents[i].numberOfLayerPairs = hostEvents[i].doublets.size();
        for (unsigned int j = 0; j < maxNumberOfLayerPairs; ++j)
        {
            auto doubletIdx = i * maxNumberOfLayerPairs + j;
            h_doublets[doubletIdx].size = 0;
        }

        for (unsigned int j = 0; j < maxNumberOfLayers; ++j)
        {
            auto layerIdx = i * maxNumberOfLayers + j;

            h_layers[layerIdx].size = 0;

        }

        for (unsigned int j = 0; j < hostEvents[i].doublets.size(); ++j)
        {
            auto layerPairIndex = i * maxNumberOfLayerPairs + j;

            h_doublets[layerPairIndex].size = hostEvents[i].doublets[j].size;
            h_doublets[layerPairIndex].innerLayerId = hostEvents[i].doublets[j].innerLayerId;
            h_doublets[layerPairIndex].outerLayerId = hostEvents[i].doublets[j].outerLayerId;

            for (unsigned int l = 0; l < hostEvents[i].rootLayers.size(); ++l)
            {
                if (hostEvents[i].rootLayers[l] == h_doublets[layerPairIndex].innerLayerId)
                {
                    auto rootlayerPairId = i * maxNumberOfRootLayerPairs
                            + h_allEvents[i].numberOfRootLayerPairs;
                    h_rootLayerPairs[rootlayerPairId] = j;
                    h_allEvents[i].numberOfRootLayerPairs++;
                }

            }
            for (unsigned int l = 0; l < hostEvents[i].doublets[j].size; ++l)
            {
                auto hitId = layerPairIndex * maxNumberOfDoublets * 2 + 2 * l;
                h_indices[hitId] = hostEvents[i].doublets[j].indices[2 * l];
                h_indices[hitId + 1] = hostEvents[i].doublets[j].indices[2 * l + 1];

            }

        }

        for (unsigned int j = 0; j < hostEvents[i].hitsLayers.size(); ++j)
        {
            auto layerIdx = i * maxNumberOfLayers + j;

            h_layers[layerIdx].size = hostEvents[i].hitsLayers[j].size;
            h_layers[layerIdx].layerId = hostEvents[i].hitsLayers[j].layerId;
            for (unsigned int l = 0; l < hostEvents[i].hitsLayers[j].size; ++l)
            {
                auto hitId = layerIdx * maxNumberOfHits + l;

                h_x[hitId] = hostEvents[i].hitsLayers[j].x[l];
                h_y[hitId] = hostEvents[i].hitsLayers[j].y[l];
                h_z[hitId] = hostEvents[i].hitsLayers[j].z[l];

            }

        }

    }

#ifdef FP_DEBUG
    for (unsigned int i = 0; i < nEvents; ++i)
    {
        assert(h_allEvents[i].eventId == hostEvents[i].eventId);
        assert(h_allEvents[i].numberOfLayers == hostEvents[i].hitsLayers.size());
        assert(h_allEvents[i].numberOfLayerPairs == hostEvents[i].doublets.size());
        auto cellId=0;
        for (unsigned int j = 0; j < hostEvents[i].doublets.size();++j)
        {
            auto layerPairIdx = i*maxNumberOfLayerPairs+j;
            assert(h_doublets[layerPairIdx].size == hostEvents[i].doublets[j].size);

            for(unsigned int l = 0; l < hostEvents[i].doublets[j].size; ++l)
            {
                auto hitId = layerPairIdx*maxNumberOfDoublets*2 + 2*l;
                assert(hostEvents[i].doublets[j].indices[2*l] == h_indices[hitId]);
                assert(hostEvents[i].doublets[j].indices[2*l+1] == h_indices[hitId+1]);
                auto innerHitId = hostEvents[i].doublets[j].indices[2*l];
                auto outerHitId = hostEvents[i].doublets[j].indices[2*l+1];
                auto innerLayerId = hostEvents[i].doublets[j].innerLayerId;
                auto outerLayerId = hostEvents[i].doublets[j].outerLayerId;

                auto xinnerIndex = i*maxNumberOfLayers*maxNumberOfHits + maxNumberOfHits*innerLayerId + innerHitId;
                assert(hostEvents[i].hitsLayers[innerLayerId].x[innerHitId]==h_x[xinnerIndex] );
//                float x1, y1, z1, x2, y2, z2;
//
//                x1 = hostEvents[i].hitsLayers[innerLayerId].x[innerHitId];
//                y1 = hostEvents[i].hitsLayers[innerLayerId].y[innerHitId];
//                z1 = hostEvents[i].hitsLayers[innerLayerId].z[innerHitId];
//                x2 = hostEvents[i].hitsLayers[outerLayerId].x[outerHitId];
//                y2 = hostEvents[i].hitsLayers[outerLayerId].y[outerHitId];
//                z2 = hostEvents[i].hitsLayers[outerLayerId].z[outerHitId];
//                printf("\n\n\nCPU cellid %d innerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n",cellId, x1,y1,z1,x2,y2,z2);

                cellId++;
            }

        }

    }
#endif

    int nGPUs;

    cudaGetDeviceCount(&nGPUs);
    std::cout << "Number of available GPUs " << nGPUs << std::endl;
    std::cout << "Using " << numberOfCPUThreads << " CPU threads " << std::endl;

    omp_set_num_threads(numberOfCPUThreads);
    unsigned int numberOfCPUOnlyThreads = numberOfCPUThreads - nGPUs;
    // HOST WORKER ALLOCATIONS

    std::vector<std::vector<GPUCACell> > hostWorker_theCells;
    hostWorker_theCells.resize(numberOfCPUOnlyThreads);
    std::vector<std::vector<GPUSimpleVector<maxCellsPerHit, unsigned int> > > hostWorker_isOuterHitOfCell;
    hostWorker_isOuterHitOfCell.resize(numberOfCPUOnlyThreads);
    std::vector<GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> > hostWorker_foundNtuplets;
    hostWorker_foundNtuplets.resize(numberOfCPUOnlyThreads);




    for(int i = 0; i <numberOfCPUOnlyThreads;++i )
    {
        hostWorker_theCells[i].resize(maxNumberOfLayerPairs * maxNumberOfDoublets);
        hostWorker_isOuterHitOfCell[i].resize(maxNumberOfLayers * maxNumberOfHits);



    }


    //GPU ALLOCATIONS
    std::cout << "preallocating memory on GPU " << std::endl;

    std::vector<Region*> d_regionParams;
    d_regionParams.resize(nGPUs);
    std::vector<GPUEvent*> d_events;
    d_events.resize(nGPUs);
    std::vector<unsigned int*> d_indices;
    d_indices.resize(nGPUs);
    std::vector<GPULayerDoublets*> d_doublets;
    d_doublets.resize(nGPUs);
    std::vector<GPULayerHits*> d_layers;
    d_layers.resize(nGPUs);
    std::vector<float*> d_x;
    d_x.resize(nGPUs);
    std::vector<float*> d_y;
    d_y.resize(nGPUs);
    std::vector<float*> d_z;
    d_z.resize(nGPUs);
    std::vector<unsigned int*> d_rootLayerPairs;
    d_rootLayerPairs.resize(nGPUs);
    std::vector < std::vector < cudaStream_t >> streams;
    streams.resize(nGPUs);
    std::vector<GPUCACell*> device_theCells;
    device_theCells.resize(nGPUs);
    std::vector<GPUSimpleVector<maxCellsPerHit, unsigned int>*> device_isOuterHitOfCell;
    device_isOuterHitOfCell.resize(nGPUs);
    std::vector<GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *> d_foundNtuplets;
    d_foundNtuplets.resize(nGPUs);




    for (unsigned int gpuIndex = 0; gpuIndex < nGPUs; ++gpuIndex)
    {
        cudaSetDevice(gpuIndex);

        cudaMalloc(&d_regionParams[gpuIndex], sizeof(Region));
        cudaMemcpy(d_regionParams[gpuIndex], h_regionParams, sizeof(Region),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_events[gpuIndex], eventsPreallocatedOnGPU * sizeof(GPUEvent));
        cudaMalloc(&d_indices[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayerPairs * maxNumberOfDoublets * 2
                        * sizeof(int));
        cudaMalloc(&d_doublets[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
        cudaMalloc(&d_layers[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayers * sizeof(GPULayerHits));
        cudaMalloc(&d_x[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayers * maxNumberOfHits * sizeof(float));
        cudaMalloc(&d_y[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayers * maxNumberOfHits * sizeof(float));
        cudaMalloc(&d_z[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayers * maxNumberOfHits * sizeof(float));
        cudaMalloc(&d_rootLayerPairs[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfRootLayerPairs * sizeof(unsigned int));
        //////////////////////////////////////////////////////////
        // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
        //////////////////////////////////////////////////////////

        cudaMalloc(&device_theCells[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayerPairs * maxNumberOfDoublets
                        * sizeof(GPUCACell));

        cudaMalloc(&device_isOuterHitOfCell[gpuIndex],
                eventsPreallocatedOnGPU * maxNumberOfLayers * maxNumberOfHits
                        * sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int> ));

        cudaMemset(device_isOuterHitOfCell[gpuIndex], 0,
                eventsPreallocatedOnGPU * maxNumberOfLayers * maxNumberOfHits
                        * sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int> ));
        //////////////////////////////////////////////////////////
        // ALLOCATIONS FOR THE RESULTS
        //////////////////////////////////////////////////////////

        cudaMalloc(&d_foundNtuplets[gpuIndex],
                eventsPreallocatedOnGPU
                        * sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));



        streams[gpuIndex].resize(numberOfCUDAStreams);
        for (int i = 0; i < numberOfCUDAStreams; ++i)
        {
            cudaStreamCreate (&streams[gpuIndex][i]);
        }

    }

    //INITIALIZATION IS NOW OVER
    //HERE STARTS THE COMPUTATION

    tbb::concurrent_queue<unsigned int> queue;
    for (unsigned iteration = 0; iteration < numberOfIterations; iteration++)
    {
        for (unsigned int i = 0; i < nEvents; ++i)
        {
            queue.push(i);
        }
    }

std::vector<unsigned int> processedEventsPerThread;
processedEventsPerThread.resize(numberOfCPUThreads,0);

double start = omp_get_wtime();
#pragma omp parallel
    {
        unsigned int streamIndex = 0;
        int threadId = omp_get_thread_num();
        unsigned int gpuIndex;
        bool isGPUThread = false;
        if (threadId < nGPUs)
        {
            gpuIndex = threadId;
            isGPUThread = true;
        }


        if (isGPUThread)
        {
            while (!queue.empty())
            {

                cudaSetDevice(gpuIndex);
                streamIndex = (streamIndex + 1) % numberOfCUDAStreams;
                cudaStreamSynchronize (streams[gpuIndex][streamIndex]);
                unsigned int i;
                queue.try_pop(i);
                processedEventsPerThread[threadId]++;

                auto d_firstLayerPairInEvt = maxNumberOfLayerPairs * streamIndex;
                auto d_firstLayerInEvt = maxNumberOfLayers * streamIndex;
                auto d_firstDoubletInEvent = d_firstLayerPairInEvt * maxNumberOfDoublets;
                auto d_firstHitInEvent = d_firstLayerInEvt * maxNumberOfHits;

                auto h_firstLayerPairInEvt = maxNumberOfLayerPairs * i;
                auto h_firstLayerInEvt = maxNumberOfLayers * i;
                auto h_firstDoubletInEvent = h_firstLayerPairInEvt * maxNumberOfDoublets;
                auto h_firstHitInEvent = h_firstLayerInEvt * maxNumberOfHits;

                for (unsigned int j = 0; j < h_allEvents[i].numberOfLayerPairs; ++j)
                {
                    h_doublets[h_firstLayerPairInEvt + j].indices =
                            &d_indices[gpuIndex][d_firstDoubletInEvent * 2
                                    + j * maxNumberOfDoublets * 2];
                    cudaMemcpyAsync(
                            &d_indices[gpuIndex][d_firstDoubletInEvent * 2
                                    + j * maxNumberOfDoublets * 2],
                            &h_indices[h_firstDoubletInEvent * 2 + j * maxNumberOfDoublets * 2],
                            h_doublets[h_firstLayerPairInEvt + j].size * 2 * sizeof(int),
                            cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                }

                for (unsigned int j = 0; j < h_allEvents[i].numberOfLayers; ++j)
                {
                    h_layers[h_firstLayerInEvt + j].x = &d_x[gpuIndex][d_firstHitInEvent
                            + maxNumberOfHits * j];
                    h_layers[h_firstLayerInEvt + j].y = &d_y[gpuIndex][d_firstHitInEvent
                            + maxNumberOfHits * j];
                    h_layers[h_firstLayerInEvt + j].z = &d_z[gpuIndex][d_firstHitInEvent
                            + maxNumberOfHits * j];
                    cudaMemcpyAsync(h_layers[h_firstLayerInEvt + j].x,
                            &h_x[h_firstHitInEvent + j * maxNumberOfHits],
                            h_layers[h_firstLayerInEvt + j].size * sizeof(float),
                            cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                    cudaMemcpyAsync(h_layers[h_firstLayerInEvt + j].y,
                            &h_y[h_firstHitInEvent + j * maxNumberOfHits],
                            h_layers[h_firstLayerInEvt + j].size * sizeof(float),
                            cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                    cudaMemcpyAsync(h_layers[h_firstLayerInEvt + j].z,
                            &h_z[h_firstHitInEvent + j * maxNumberOfHits],
                            h_layers[h_firstLayerInEvt + j].size * sizeof(float),
                            cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                }

                cudaMemcpyAsync(
                        &d_rootLayerPairs[gpuIndex][maxNumberOfRootLayerPairs * streamIndex],
                        &h_rootLayerPairs[maxNumberOfRootLayerPairs * i],
                        h_allEvents[i].numberOfRootLayerPairs * sizeof(unsigned int),
                        cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                cudaMemcpyAsync(&d_doublets[gpuIndex][d_firstLayerPairInEvt],
                        &h_doublets[h_firstLayerPairInEvt],
                        h_allEvents[i].numberOfLayerPairs * sizeof(GPULayerDoublets),
                        cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);
                cudaMemcpyAsync(&d_layers[gpuIndex][d_firstLayerInEvt],
                        &h_layers[h_firstLayerInEvt],
                        h_allEvents[i].numberOfLayers * sizeof(GPULayerHits),
                        cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);

                cudaMemcpyAsync(&d_events[gpuIndex][streamIndex], &h_allEvents[i], sizeof(GPUEvent),
                        cudaMemcpyHostToDevice, streams[gpuIndex][streamIndex]);

                dim3 numberOfBlocks_create(32, h_allEvents[i].numberOfLayerPairs);
                dim3 numberOfBlocks_connect(16, h_allEvents[i].numberOfLayerPairs);
                dim3 numberOfBlocks_find(8, h_allEvents[i].numberOfRootLayerPairs);
// KERNELS
//        debug_input_data<<<1,1,0,streams[streamIndex]>>>(&d_events[streamIndex], &d_doublets[d_firstLayerPairInEvt], &d_layers[d_firstLayerInEvt],d_regionParams,  maxNumberOfHits );
                kernel_create<<<numberOfBlocks_create,256,0,streams[gpuIndex][streamIndex]>>>(&d_events[gpuIndex][streamIndex], &d_doublets[gpuIndex][d_firstLayerPairInEvt],
                        &d_layers[gpuIndex][d_firstLayerInEvt], &device_theCells[gpuIndex][d_firstLayerPairInEvt*maxNumberOfDoublets],
                        &device_isOuterHitOfCell[gpuIndex][d_firstHitInEvent], &d_foundNtuplets[gpuIndex][streamIndex],d_regionParams[gpuIndex], maxNumberOfDoublets, maxNumberOfHits);

////
//        kernel_debug<<<1,1,0,streams[streamIndex]>>>(&d_events[streamIndex], &d_doublets[d_firstLayerPairInEvt],
//                &d_layers[d_firstLayerInEvt], &device_theCells[d_firstLayerPairInEvt*maxNumberOfDoublets],
//                &device_isOuterHitOfCell[d_firstHitInEvent], &d_foundNtuplets[streamIndex],
//                d_regionParams, theThetaCut, thePhiCut,theHardPtCut,maxNumberOfDoublets, maxNumberOfHits);
                kernel_connect<<<numberOfBlocks_connect,256,0,streams[gpuIndex][streamIndex]>>>(&d_events[gpuIndex][streamIndex],
                        &d_doublets[gpuIndex][d_firstLayerPairInEvt], &device_theCells[gpuIndex][d_firstLayerPairInEvt*maxNumberOfDoublets],
                        &device_isOuterHitOfCell[gpuIndex][d_firstHitInEvent], d_regionParams[gpuIndex], theThetaCut, thePhiCut,
                        theHardPtCut, maxNumberOfDoublets, maxNumberOfHits);

//        kernel_debug_connect<<<1,1,0,streams[streamIndex]>>>(&d_events[streamIndex], &d_doublets[d_firstLayerPairInEvt],
//                &device_theCells[d_firstLayerPairInEvt*maxNumberOfDoublets], &device_isOuterHitOfCell[d_firstHitInEvent],
//                 d_regionParams, maxNumberOfDoublets, maxNumberOfHits);
//        cudaMemsetAsync(&d_foundNtuplets[streamIndex], 0, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ), streams[streamIndex]);

                kernel_find_ntuplets<<<numberOfBlocks_find,256,0,streams[gpuIndex][streamIndex]>>>(&d_events[gpuIndex][streamIndex],
                        &d_doublets[gpuIndex][d_firstLayerPairInEvt], &device_theCells[gpuIndex][d_firstLayerPairInEvt*maxNumberOfDoublets],
                        &d_foundNtuplets[gpuIndex][streamIndex],&d_rootLayerPairs[gpuIndex][maxNumberOfRootLayerPairs*streamIndex], 4 , maxNumberOfDoublets);

//        kernel_debug_find_ntuplets<<<1,1,0,streams[streamIndex]>>>(&d_events[streamIndex],
//                &d_doublets[d_firstLayerPairInEvt], &device_theCells[d_firstLayerPairInEvt*maxNumberOfDoublets],
//                &d_foundNtuplets[streamIndex],&d_rootLayerPairs[maxNumberOfRootLayerPairs*streamIndex], 4 , maxNumberOfDoublets);
                cudaMemcpyAsync(&h_foundNtuplets[streamIndex],
                        &d_foundNtuplets[gpuIndex][streamIndex],
                        sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ),
                        cudaMemcpyDeviceToHost, streams[gpuIndex][streamIndex]);
                cudaMemsetAsync(&device_isOuterHitOfCell[gpuIndex][d_firstHitInEvent], 0,
                        maxNumberOfLayers * maxNumberOfHits
                                * sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int> ),
                        streams[gpuIndex][streamIndex]);

//        cudaStreamSynchronize(streams[streamIndex]);
//        std::cout << "found quadruplets " << h_foundNtuplets[streamIndex].size() << std::endl;

            }

            for (int i = 0; i < numberOfCUDAStreams; ++i)
            {
                cudaStreamSynchronize (streams[gpuIndex][i]);
            }

        }
        else
        {
            int CPUOnlyThreadId = threadId - nGPUs;

            while (!queue.empty())
            {
                unsigned int i;
                queue.try_pop(i);
                processedEventsPerThread[threadId]++;

                auto h_firstLayerPairInEvt = maxNumberOfLayerPairs * i;
                auto h_firstLayerInEvt = maxNumberOfLayers * i;
                auto h_firstDoubletInEvent = h_firstLayerPairInEvt * maxNumberOfDoublets;
                auto h_firstHitInEvent = h_firstLayerInEvt * maxNumberOfHits;
                std::vector<GPULayerDoublets> doublets;
                doublets.resize(h_allEvents[i].numberOfLayerPairs);


                for (unsigned int j = 0; j < h_allEvents[i].numberOfLayerPairs; ++j)
                {
                    doublets[j] = h_doublets[h_firstLayerPairInEvt + j];
                    doublets[j].indices =&h_indices[h_firstDoubletInEvent* 2 + j * maxNumberOfDoublets * 2];

                }

                std::vector<GPULayerHits> layerHits;
                layerHits.resize(h_allEvents[i].numberOfLayers);
                for (unsigned int j = 0; j < h_allEvents[i].numberOfLayers; ++j)
                {
                    layerHits[j] = h_layers[h_firstLayerInEvt + j];
                    layerHits[j].x=&h_x[h_firstHitInEvent + j * maxNumberOfHits];
                    layerHits[j].y=&h_y[h_firstHitInEvent + j * maxNumberOfHits];
                    layerHits[j].z=&h_z[h_firstHitInEvent + j * maxNumberOfHits];

                }

                host_kernel(&h_allEvents[i], doublets.data(),
                        layerHits.data(),
                        hostWorker_theCells[CPUOnlyThreadId],
                        hostWorker_isOuterHitOfCell[CPUOnlyThreadId], &h_rootLayerPairs[maxNumberOfRootLayerPairs * i], &hostWorker_foundNtuplets[CPUOnlyThreadId],
                        h_regionParams, theThetaCut, thePhiCut, theHardPtCut, maxNumberOfDoublets,
                        maxNumberOfHits);
                for(unsigned int j = 0; j < hostWorker_isOuterHitOfCell[CPUOnlyThreadId].size(); ++j)
                {
                    hostWorker_isOuterHitOfCell[CPUOnlyThreadId][j].reset();
                }

            }
        }

    }


double stop = omp_get_wtime();

    std::cout << "Summary: " << std::endl;
    unsigned int processedByGPU = 0;
    unsigned int processedByCPU = 0;

    for(unsigned int i = 0; i< processedEventsPerThread.size(); ++i)
    {
        std::cout << "\tthread " << i << " processed " << processedEventsPerThread[i] << " events." << std::endl;

        if(i < nGPUs)
            processedByGPU+=processedEventsPerThread[i];
        else
            processedByCPU+=processedEventsPerThread[i];
    }

    std::cout << numberOfIterations*nEvents << " events processed in " << stop-start << "s. Measured rate: " << numberOfIterations*nEvents/(stop-start) << " Hz " << std::endl;
    std::cout << processedByGPU << " events processed by " << nGPUs << " GPUs in  " << stop-start << "s. Measured GPU rate: " << processedByGPU/(stop-start) << " Hz " << std::endl;
    std::cout << processedByCPU << " events processed by " << numberOfCPUOnlyThreads << " CPUs in  " << stop-start << "s. Measured CPU rate: " << processedByCPU/(stop-start) << " Hz " << std::endl;

// CLEANUP


    std::cout << "deleting Device memory " << std::endl;

    for (unsigned int gpuIndex = 0; gpuIndex < nGPUs; ++gpuIndex)
    {
        cudaSetDevice(gpuIndex);
        for (int i = 0; i < numberOfCUDAStreams; ++i)
        {
            cudaStreamSynchronize (streams[gpuIndex][i]);

            cudaStreamDestroy(streams[gpuIndex][i]);

        }

        cudaFree(device_isOuterHitOfCell[gpuIndex]);
        cudaFree(d_foundNtuplets[gpuIndex]);

        cudaFree(d_regionParams[gpuIndex]);
        cudaFree(device_theCells[gpuIndex]);

        cudaFree(d_events[gpuIndex]);
        cudaFree(d_indices[gpuIndex]);
        cudaFree(d_doublets[gpuIndex]);
        cudaFree(d_layers[gpuIndex]);
        cudaFree(d_x[gpuIndex]);
        cudaFree(d_y[gpuIndex]);
        cudaFree(d_z[gpuIndex]);
        cudaFree(d_rootLayerPairs[gpuIndex]);

    }
    std::cout << "deleting Host memory " << std::endl;

    cudaFreeHost(h_foundNtuplets);
    cudaFreeHost(h_regionParams);
    cudaFreeHost(h_allEvents);
    cudaFreeHost(h_layers);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);
    cudaFreeHost(h_rootLayerPairs);
    cudaFreeHost(h_indices);
    cudaFreeHost(h_doublets);

    return 0;
}

