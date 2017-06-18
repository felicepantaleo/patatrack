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
