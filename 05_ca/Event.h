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
#pragma once

#include <vector>
#include "GPUHitsAndDoublets.h"

struct HostEvent
{
    unsigned int eventId;
    std::vector<int> rootLayers;
    std::vector<HostLayerHits> hitsLayers;
    std::vector<HostLayerDoublets> doublets;

};


struct GPUEvent
{
    unsigned int eventId;
    unsigned int numberOfRootLayerPairs;
    unsigned int numberOfLayers;
    unsigned int numberOfLayerPairs;

};

struct Region
{
        float ptmin, region_origin_x, region_origin_y , region_origin_radius;
};
