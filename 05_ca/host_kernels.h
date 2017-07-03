//   Copyright 2017, Felice Pantaleo, CERN
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

template<int maxNumberOfQuadruplets>
void host_kernel(const GPUEvent* event, const GPULayerDoublets* gpuDoublets,
        const GPULayerHits* gpuHitsOnLayers, std::vector<GPUCACell>& cells,
        std::vector<GPUSimpleVector<100, unsigned int> >& isOuterHitOfCell, unsigned int* rootLayerPairs,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets, const Region* region,
        const float thetaCut, const float phiCut, const float hardPtCut,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{

    unsigned int numberOfLayerPairs = event->numberOfLayerPairs;
    unsigned int numberOfLayers = event->numberOfLayers;
    foundNtuplets->reset();
    for(unsigned int i = 0; i< maxNumberOfHits; ++i)
        isOuterHitOfCell[i].reset();
//    printf("kernel_debug_create: theEvent contains numberOfLayerPairs: %d\n", numberOfLayerPairs);
    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; ++layerPairIndex)
    {

        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        int numberOfDoublets = gpuDoublets[layerPairIndex].size;
//        printf("kernel_debug_create: layerPairIndex: %d inner %d outer %d size %u\n",
//                layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

        auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
        auto globalFirstHitIdx = outerLayerId * maxNumberOfHits;
//        printf("kernel_debug_create: theIdOfThefirstCellInLayerPair: %d globalFirstHitIdx %d\n",
//                globalFirstDoubletIdx, globalFirstHitIdx);

        for (unsigned int i = 0; i < gpuDoublets[layerPairIndex].size; i++)
        {

            auto globalCellIdx = i + globalFirstDoubletIdx;
            auto& thisCell = cells[globalCellIdx];
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex,
                    globalCellIdx, gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region->region_origin_x, region->region_origin_y);

            isOuterHitOfCell.at(globalFirstHitIdx + outerHitId).push_back(globalCellIdx);
        }

    }

    //starting connect

    float ptmin, region_origin_x, region_origin_y, region_origin_radius;

    ptmin = region->ptmin;
    region_origin_x = region->region_origin_x;
    region_origin_y = region->region_origin_y;
    region_origin_radius = region->region_origin_radius;

    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; ++layerPairIndex)
    {

//        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        int numberOfDoublets = gpuDoublets[layerPairIndex].size;
//        printf("kernel_debug_connect: connecting layerPairIndex: %d inner %d outer %d size %u\n",
//                layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

        auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
        auto globalFirstHitIdx = innerLayerId * maxNumberOfHits;
//        printf("kernel_debug_connect: theIdOfThefirstCellInLayerPair: %d globalFirstHitIdx %d\n", globalFirstDoubletIdx, globalFirstHitIdx);

        for (unsigned int i = 0; i < numberOfDoublets; i++)
        {

            auto globalCellIdx = i + globalFirstDoubletIdx;

             auto& thisCell = cells[globalCellIdx];
             auto innerHitId = thisCell.get_inner_hit_id();
             auto numberOfPossibleNeighbors =
                     isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
             for (auto j = 0; j < numberOfPossibleNeighbors; ++j)
             {
                 unsigned int otherCell = isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

                 if (thisCell.check_alignment_and_tag(cells.data(), otherCell, ptmin, region_origin_x,
                         region_origin_y, region_origin_radius, thetaCut, phiCut, hardPtCut))
                 {
//                     printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n", globalCellIdx, otherCell);

                     cells.at(otherCell).theOuterNeighbors.push_back(globalCellIdx);
                 }
             }
        }

    }

    auto numberOfRootLayerPairs =event->numberOfRootLayerPairs;
//    printf("numberOfRootLayerPairs = %d\n", numberOfRootLayerPairs);

    for(int rootLayerPair = 0; rootLayerPair < numberOfRootLayerPairs; ++rootLayerPair)
    {
        unsigned int rootLayerPairIndex = rootLayerPairs[rootLayerPair];


        auto globalFirstDoubletIdx = rootLayerPairIndex*maxNumberOfDoublets;

        GPUSimpleVector<3, unsigned int> stack;
//        printf("rootLayerPair = %d number of doublets %d\n", rootLayerPairIndex, gpuDoublets[rootLayerPairIndex].size);

        for (int i =0; i < gpuDoublets[rootLayerPairIndex].size; i++)
        {

            auto globalCellIdx = i+globalFirstDoubletIdx;

            stack.reset();
            stack.push_back(globalCellIdx);
            cells[globalCellIdx].find_ntuplets_host(cells.data(), foundNtuplets, stack, 4);

        }
    }
}
