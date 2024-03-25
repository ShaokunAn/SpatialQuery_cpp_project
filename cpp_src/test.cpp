#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::pair<py::dict, py::dict>
SpatialDataMultiple::differentialAnalysisKnn(const std::string &cellType,
                                             const py::list &datasets,
                                             const py::list &fovsID1,
                                             const py::list &fovsID2,
                                             int k = 30,
                                             int minCount = 0,
                                             float minSupport = 0.5)
{
    // Identify frequent patterns in each dataset
    std::vector<std::unordered_map<std::string, std::vector<double>>> fpDatasets(2);
    std::vector<std::unordered_set<std::string>> itemsets(2);

    for (size_t i = 0; i < 2; ++i)
    {
        const py::list &fovsID = (i == 0) ? fovsID1 : fovsID2;
        std::string datasetName = datasets[i].cast<std::string>();

        std::unordered_map<std::string, std::unordered_map<std::string, double>> patternSupport;
        for (const auto &fovID : fovsID)
        {
            int fov = py::cast<int>(fovID);
            if (fovs.find(fov) == fovs.end())
            {
                throw std::invalid_argument("Invalid fovID: " + std::to_string(fov));
            }

            py::dict fpFov = findFPKnnFov(cellType, fovs[fov], k, minCount, minSupport);
            if (fpFov.empty())
            {
                continue;
            }

            for (const auto &item : fpFov)
            {
                std::string itemset = item.first.cast<std::string>();
                double support = item.second.cast<double>();
                patternSupport[itemset][std::to_string(fov)] = support;
            }
        }

        py::dict fpDataset;
        fpDataset["itemsets"] = py::cast(patternSupport.keys());
        for (const auto &[itemset, fovSupportMap] : patternSupport)
        {
            std::vector<double> supportValues;
            for (const auto &fovID : fovsID)
            {
                int fov = py::cast<int>(fovID);
                std::string fovName = std::to_string(fov);
                if (fovSupportMap.find(fovName) != fovSupportMap.end())
                {
                    supportValues.push_back(fovSupportMap.at(fovName));
                }
                else
                {
                    supportValues.push_back(0.0);
                }
            }
            fpDataset["support_" + datasetName + "_" + itemset] = py::cast(supportValues);
        }

        fpDatasets[datasetName] = fpDataset;
    }

    return {fpDatasets[datasets[0].cast<std::string>()], fpDatasets[datasets[1].cast<std::string>()]};
}