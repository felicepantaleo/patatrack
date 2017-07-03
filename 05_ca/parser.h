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

#ifndef GPUTT_PARSER_H_
#define GPUTT_PARSER_H_

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include "Event.h"

void parseinputFile(std::string inputFile, std::vector<HostEvent>& OutputEvents, Region& region,
        unsigned int maxEvents, unsigned int& maxNumberOfHits, unsigned int& maxNumberOfDoublets )
{

    char line[1024];
    std::ifstream inFile(inputFile.c_str(), std::ifstream::in);
    if (!inFile)
    {
        std::cout << "could not open txt file " << inputFile << std::endl;
        exit(1);
    }
    HostEvent tmpEvent;



    bool finishedParsing = false;
    int lineNumber = 0;
    while (inFile.good() && !finishedParsing)
    {
        lineNumber++;
        inFile.getline(line, 1023);
        std::string stringline(line);
        if (stringline.empty())
            continue;
        std::stringstream ss;
        ss.str(stringline);
        std::string title;
        std::string PARSER_title;

        std::string uselessText;

        ss >> PARSER_title >> title;


        if (title.compare("NewCellularAutomatonStarting") == 0)
        {
            tmpEvent.eventId = OutputEvents.size();
            std::cout << " parsing new event " << std::endl;
            std::cout << " maxEvents: " << maxEvents << " number of events " << OutputEvents.size()<< std::endl;
            if (maxEvents > 0 && OutputEvents.size() == maxEvents)
            {
                finishedParsing = true;
            }
            else
            {

                OutputEvents.emplace_back(tmpEvent);

            }
        }
        else if (title.compare("Layer") == 0)
        {
            auto& event = OutputEvents.back();
            HostLayerHits tmpLayerHits;
            ss >> tmpLayerHits.layerId;
            event.hitsLayers.push_back(tmpLayerHits);

        }
        else if (title.compare("numberOfHitsOnLayer") == 0)
        {

            auto& event = OutputEvents.back();
            unsigned int size;
            ss >> size;
            std::cout << "parsing numberhitsOnLayer " << size << std::endl;

            event.hitsLayers.back().size = size;
            event.hitsLayers.back().x.reserve(size);
            event.hitsLayers.back().y.reserve(size);
            event.hitsLayers.back().z.reserve(size);
            maxNumberOfHits = std::max(maxNumberOfHits,size );

        }
        else if (title.compare("hit") == 0)
        {
            float x, y, z;
            ss >> uselessText >> x >> y >> z;

//            std::cout << "parsing hit " << uselessText << " " << x << " " << y << " " << z << std::endl;
            auto& event = OutputEvents.back();
            event.hitsLayers.back().x.push_back(x);
            event.hitsLayers.back().y.push_back(y);
            event.hitsLayers.back().z.push_back(z);
        }

        else if (title.compare("ptmin") == 0)
        {
            ss >> region.ptmin >> uselessText >> region.region_origin_x >> uselessText
                    >> region.region_origin_y >> uselessText >> region.region_origin_radius;

        }
        else if (title.compare("numberOfRootLayers") == 0)
        {

            auto& event = OutputEvents.back();
            int numberOfRootLayers;
            ss >> numberOfRootLayers;
            event.rootLayers.reserve(numberOfRootLayers);

        }

        else if (title.compare("RootLayer") == 0)
        {
            auto& event = OutputEvents.back();
            int rootLayerId;
            ss >> rootLayerId;
            event.rootLayers.push_back(rootLayerId);
        }
        else if (title.compare("LayerPairId") == 0)
        {
            auto& event = OutputEvents.back();
            HostLayerDoublets tmpHostLayerDoublets;
            event.doublets.push_back(tmpHostLayerDoublets);
        }
        else if (title.compare("InnerLayerId") == 0)
        {
            auto& doublet = OutputEvents.back().doublets.back();

            int innerLayer;
            ss >> innerLayer;
            doublet.innerLayerId = innerLayer;

        }
        else if (title.compare("OuterLayerId") == 0)
        {
            auto& doublet = OutputEvents.back().doublets.back();

            int OuterLayerId;
            ss >> OuterLayerId;
            doublet.outerLayerId = OuterLayerId;

        }

        else if (title.compare("numberOfDoublets") == 0)
        {
            auto& doublet = OutputEvents.back().doublets.back();

            unsigned int numberOfDoublets;
            ss >> numberOfDoublets;
            doublet.size = numberOfDoublets;

            doublet.indices.reserve(2 * numberOfDoublets);
            maxNumberOfDoublets = std::max(maxNumberOfDoublets,numberOfDoublets );
//            std::cout << "parsing numberOfDoublets " << numberOfDoublets  << std::endl;

        }

        else if (title.compare("innerHitId") == 0 )
        {
            auto& doublet = OutputEvents.back().doublets.back();

            unsigned int hitId;
            ss >> hitId;
//            std::cout << "parsing inner hitId " << hitId  << std::endl;

            doublet.indices.push_back(hitId);

        }
        else if (title.compare("outerHitId") == 0)
        {
            auto& doublet = OutputEvents.back().doublets.back();

            unsigned int hitId;
            ss >> hitId;
//            std::cout << "parsing outer hitId " << hitId <<std::endl;

            doublet.indices.push_back(hitId);

        }

        if (ss.fail())
            exit(1);

    }
    inFile.close();
}

#endif /* GPUTT_PARSER_H_ */
