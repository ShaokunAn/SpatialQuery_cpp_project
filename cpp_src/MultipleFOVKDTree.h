#include <unordered_map>
#include "SingleFOVKDTree.h"

class SpatialDataMultiple
{
public:
    SpatialDataMultiple(){};
    // Build multiple FOV by adding FOV data to current FOV data
    void setFOVData(int fovID, const py::array_t<float> &coordinates, const py::list &labels);
    py::array_t<float> getCoordinatesFOV(int fovID);
    py::list buildFPTreeKNN(const Item &cellType, const py::list &fovIDs, int k = 30, float minSupport = 0.5, bool disDuplicates = false);
    py::list buildFPTreeDist(const Item &cellType, const py::list &fovIDs, float radius = 100.0, float minSupport = 0.5, bool disDuplicates = false, int minSize = 0);
    py::list motifEnrichmentKNN(const Item &cellType, const py::list &motifs, py::list &fovIDs, int k = 30, float minSupport = 0.5, bool disDuplicates = false, float maxDist = 200.0);
    py::list motifEnrichmentDist(const Item &cellType, const py::list &motifs, py::list &fovIDs, float radius = 100.0, float minSupport = 0.5, bool disDuplicates = false, int minSize = 0);
    py::list findFPKnnFOV(const std::string &cellType, const int fovID, int k = 30,
                          float minSupport = 0.5);
    py::list findFPDistFOV(const std::string &cellType, const int fovID, float maxDist = 100.0,
                           int minSize = 0, float minSupport = 0.5);
    py::tuple differentialAnalysisKnn(const std::string &cellType,
                                      const py::list &fovsID1,
                                      const py::list &fovsID2,
                                      int k = 30,
                                      float minSupport = 0.5);
    py::tuple differentialAnalysisDist(const std::string &cellType,
                                       const py::list &fovsID1,
                                       const py::list &fovsID2,
                                       float maxDist = 100.0,
                                       float minSupport = 0.5,
                                       int minSize = 0);

    // private:
    std::unordered_map<int, SpatialDataSingle> fovs;
};