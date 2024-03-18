#include <unordered_map>
#include "SingleFOVKDTree.h"

class SpatialDataMultiple
{
public:
    SpatialDataMultiple() {};
    // Build multiple FOV by adding FOV data to current FOV data
    void addFOVData(int fovID, const py::array_t<float> &coordinates, const std::vector<std::string> &labels);

private:
    std::unordered_map<int, SpatialDataSingle> fovs;
}