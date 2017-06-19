#pragma once

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void debug_input_data(const GPUEvent* event, const GPULayerDoublets* gpuDoublets,
        const GPULayerHits* gpuHitsOnLayers, const Region* region, unsigned int maxNumberOfHits)
{
    float ptmin, region_origin_x, region_origin_y, region_origin_radius;

    ptmin = region->ptmin;
    region_origin_x = region->region_origin_x;
    region_origin_y = region->region_origin_y;
    region_origin_radius = region->region_origin_radius;

    printf(
            "GPU: Region ptmin %f , region_origin_x %f , region_origin_y %f , region_origin_radius  %f \n",
            ptmin, region_origin_x, region_origin_y, region_origin_radius);
    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
    printf("GPU: numberOfLayerPairs: %d\n", numberOfLayerPairs);

    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; ++layerPairIndex)
    {
        printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);
        printf("\t innerLayer: %d outerLayer: %d \n", gpuDoublets[layerPairIndex].innerLayerId,
                gpuDoublets[layerPairIndex].outerLayerId);

        for (unsigned int cellIndexInLayerPair = 0;
                cellIndexInLayerPair < gpuDoublets[layerPairIndex].size; ++cellIndexInLayerPair)
        {

            if (cellIndexInLayerPair < 5)
            {
                auto innerhit = gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair];
                auto innerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId].x[innerhit];
                auto innerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId].y[innerhit];
                auto innerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId].z[innerhit];

                auto outerhit = gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1];
                auto outerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId].x[outerhit];
                auto outerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId].y[outerhit];
                auto outerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId].z[outerhit];
                printf("\t \t %d innerHit: %d %f %f %f outerHit: %d %f %f %f\n",
                        cellIndexInLayerPair, innerhit, innerX, innerY, innerZ, outerhit, outerX,
                        outerY, outerZ);
            }
        }

    }
}

template<int maxNumberOfQuadruplets>
__global__
void kernel_create(const GPUEvent* event, const GPULayerDoublets* gpuDoublets,
        const GPULayerHits* gpuHitsOnLayers, GPUCACell* cells, GPUSimpleVector<100, unsigned int> * isOuterHitOfCell,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets, const Region* region,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{

    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if(cellIndexInLayerPair == 0 && layerPairIndex == 0)
    {
        foundNtuplets->reset();
    }

    if (layerPairIndex < numberOfLayerPairs)
    {
        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        auto globalFirstDoubletIdx = layerPairIndex*maxNumberOfDoublets;
        auto globalFirstHitIdx = outerLayerId*maxNumberOfHits;

        for (unsigned int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i+globalFirstDoubletIdx;
            auto& thisCell = cells[globalCellIdx];
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId, region->region_origin_x,  region->region_origin_y);

            isOuterHitOfCell[globalFirstHitIdx+outerHitId].push_back_ts(globalCellIdx);
        }
    }
}

template<int maxNumberOfQuadruplets>
__global__
void kernel_debug(const GPUEvent* event, const GPULayerDoublets* gpuDoublets,
        const GPULayerHits* gpuHitsOnLayers, GPUCACell* cells, GPUSimpleVector<100, unsigned int> * isOuterHitOfCell,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,const Region* region,
        const float thetaCut, const float phiCut, const float hardPtCut,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{

    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
    unsigned int numberOfLayers = event->numberOfLayers;
    printf("kernel_debug_create: theEvent contains numberOfLayerPairs: %d\n", numberOfLayerPairs);
    for(unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs;++layerPairIndex )
    {

        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        int numberOfDoublets = gpuDoublets[layerPairIndex].size;
        printf("kernel_debug_create: layerPairIndex: %d inner %d outer %d size %u\n", layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

        auto globalFirstDoubletIdx = layerPairIndex*maxNumberOfDoublets;
        auto globalFirstHitIdx = outerLayerId*maxNumberOfHits;
        printf("kernel_debug_create: theIdOfThefirstCellInLayerPair: %d globalFirstHitIdx %d\n", globalFirstDoubletIdx, globalFirstHitIdx);

        for (unsigned int i = 0; i < gpuDoublets[layerPairIndex].size;i++)
        {

            auto globalCellIdx = i+globalFirstDoubletIdx;
            auto& thisCell = cells[globalCellIdx];
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,region->region_origin_x,  region->region_origin_y);

            isOuterHitOfCell[globalFirstHitIdx+outerHitId].push_back_ts(globalCellIdx);
        }

    }

    for(unsigned int layerIndex = 0; layerIndex < numberOfLayers;++layerIndex )
    {
        auto numberOfHitsOnLayer = gpuHitsOnLayers[layerIndex].size;
        for(unsigned hitId = 0; hitId < numberOfHitsOnLayer; hitId++)
        {

            if(isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size()>0)
            {
                printf("layer %d hit %d is outer hit of %d cells\n",layerIndex, hitId, isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size());
                for(unsigned cell = 0; cell< isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size(); cell++)
                {
                    printf("cell %d\n", isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].m_data[cell]);

                }
            }

        }

    }

    //starting connect

    float ptmin, region_origin_x, region_origin_y, region_origin_radius;

    ptmin = region->ptmin;
    region_origin_x = region->region_origin_x;
    region_origin_y = region->region_origin_y;
    region_origin_radius = region->region_origin_radius;

    for(unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs;++layerPairIndex )
    {

        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        int numberOfDoublets = gpuDoublets[layerPairIndex].size;
        printf("kernel_debug_connect: connecting layerPairIndex: %d inner %d outer %d size %u\n", layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

        auto globalFirstDoubletIdx = layerPairIndex*maxNumberOfDoublets;
        auto globalFirstHitIdx = innerLayerId*maxNumberOfHits;
//        printf("kernel_debug_connect: theIdOfThefirstCellInLayerPair: %d globalFirstHitIdx %d\n", globalFirstDoubletIdx, globalFirstHitIdx);

        for (unsigned int i = 0; i < numberOfDoublets;i++)
        {

            auto globalCellIdx = i+globalFirstDoubletIdx;

            auto& thisCell = cells[globalCellIdx];
            auto innerHitId = thisCell.get_inner_hit_id();
            auto numberOfPossibleNeighbors =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
//            if(numberOfPossibleNeighbors>0)
//            printf("kernel_debug_connect: cell: %d has %d possible neighbors\n", globalCellIdx, numberOfPossibleNeighbors);
            float x1, y1, z1, x2, y2, z2;

            x1 = thisCell.get_inner_x();
            y1 = thisCell.get_inner_y();
            z1 = thisCell.get_inner_z();
            x2 = thisCell.get_outer_x();
            y2 = thisCell.get_outer_y();
            z2 = thisCell.get_outer_z();
            printf("\n\n\nDEBUG cellid %d innerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n",globalCellIdx, x1,y1,z1,x2,y2,z2);

            for (auto j = 0; j < numberOfPossibleNeighbors; ++j)
            {
                unsigned int otherCell = isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];
//                printf("kernel_debug_connect: checking compatibility with %d \n", otherCell);
//                float x3, y3, z3, x4, y4, z4;
//                x3 = cells[otherCell].get_inner_x();
//                y3 = cells[otherCell].get_inner_y();
//                z3 = cells[otherCell].get_inner_z();
//                x4 = cells[otherCell].get_outer_x();
//                y4 = cells[otherCell].get_outer_y();
//                z4 = cells[otherCell].get_outer_z();
//                printf("DEBUG \tinnerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n",x3,y3,z3,x4,y4,z4);



                if (thisCell.check_alignment_and_tag(cells, otherCell, ptmin, region_origin_x,
                                region_origin_y, region_origin_radius, thetaCut, phiCut, hardPtCut))
                {
                    printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n", globalCellIdx, otherCell);

                    cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
                }
            }
        }

    }
}

__global__
void kernel_connect(const GPUEvent* event, const GPULayerDoublets* gpuDoublets, GPUCACell* cells,
        GPUSimpleVector<100, unsigned int> * isOuterHitOfCell, const Region* region,
        const float thetaCut, const float phiCut, const float hardPtCut,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{
    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
    float ptmin, region_origin_x, region_origin_y, region_origin_radius;

    ptmin = region->ptmin;
    region_origin_x = region->region_origin_x;
    region_origin_y = region->region_origin_y;
    region_origin_radius = region->region_origin_radius;

    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if (layerPairIndex < numberOfLayerPairs)
    {
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
        auto globalFirstHitIdx = innerLayerId * maxNumberOfHits;

        for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i + globalFirstDoubletIdx;

            auto& thisCell = cells[globalCellIdx];
            auto innerHitId = thisCell.get_inner_hit_id();
            auto numberOfPossibleNeighbors =
                    isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
            for (auto j = 0; j < numberOfPossibleNeighbors; ++j)
            {
                unsigned int otherCell = isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

                if (thisCell.check_alignment_and_tag(cells, otherCell, ptmin, region_origin_x,
                        region_origin_y, region_origin_radius, thetaCut, phiCut, hardPtCut))
                {
//                    printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n", globalCellIdx, otherCell);

                    cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
                }
            }
        }
    }
}

__global__
void kernel_debug_connect(const GPUEvent* event, const GPULayerDoublets* gpuDoublets,
        GPUCACell* cells, GPUSimpleVector<100, unsigned int> * isOuterHitOfCell,
        const Region* region, unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{
    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
//    float ptmin, region_origin_x, region_origin_y , region_origin_radius;
//
//    ptmin = region->ptmin;
//    region_origin_x = region->region_origin_x;
//    region_origin_y = region->region_origin_y;
//    region_origin_radius = region->region_origin_radius;
    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; layerPairIndex++)
    {
        if (layerPairIndex < numberOfLayerPairs)
        {
//            int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
            auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
            for (int i = 0; i < gpuDoublets[layerPairIndex].size; i++)
            {
                auto globalCellIdx = i + globalFirstDoubletIdx;

                auto& thisCell = cells[globalCellIdx];
                if (thisCell.theOuterNeighbors.size() > 0)
                {
                    thisCell.print_cell();
//                    thisCell.print_neighbors();
                }
            }
        }
    }

}

template<int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(const GPUEvent* event,
        const GPULayerDoublets* gpuDoublets,
        GPUCACell* cells,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
        unsigned int* rootLayerPairs,
        unsigned int minHitsPerNtuplet, unsigned int maxNumberOfDoublets)
{
    auto numberOfRootLayerPairs =event->numberOfRootLayerPairs;
//    printf("numberOfRootLayerPairs = %d", numberOfRootLayerPairs);
    if(blockIdx.y < numberOfRootLayerPairs)
    {
        unsigned int cellIndexInRootLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];
        auto globalFirstDoubletIdx = rootLayerPairIndex*maxNumberOfDoublets;

        GPUSimpleVector<3, unsigned int> stack;
        for (int i = cellIndexInRootLayerPair; i < gpuDoublets[rootLayerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i+globalFirstDoubletIdx;
            stack.reset();
            stack.push_back(globalCellIdx);
            cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);

        }
    }
//    printf("found quadruplets: %d", foundNtuplets->size());

}

template<int maxNumberOfQuadruplets>
__global__
void kernel_debug_find_ntuplets(const GPUEvent* event,
        const GPULayerDoublets* gpuDoublets,
        GPUCACell* cells,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
        unsigned int* rootLayerPairs,
        unsigned int minHitsPerNtuplet, unsigned int maxNumberOfDoublets)
{
    auto numberOfRootLayerPairs =event->numberOfRootLayerPairs;
    printf("numberOfRootLayerPairs = %d", numberOfRootLayerPairs);
    for(int rootLayerPair = 0; rootLayerPair < numberOfRootLayerPairs; ++rootLayerPair)
    {
        unsigned int rootLayerPairIndex = rootLayerPairs[rootLayerPair];
        auto globalFirstDoubletIdx = rootLayerPairIndex*maxNumberOfDoublets;

        GPUSimpleVector<3, unsigned int> stack;
        for (int i =0; i < gpuDoublets[rootLayerPairIndex].size; i++)
        {
            auto globalCellIdx = i+globalFirstDoubletIdx;
            stack.reset();
            stack.push_back(globalCellIdx);
            cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);

        }
        printf("found quadruplets: %d", foundNtuplets->size());
    }
}


