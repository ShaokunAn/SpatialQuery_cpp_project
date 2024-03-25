#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_set>
#include "typedef.h"


namespace py = pybind11;

class SpatialDataSingle
{
public:
    py::array_t<float> coordinates;
    std::vector<std::string> labels;
    pcl::KdTreeFLANN<pcl::PointXY> kdtree;

    SpatialDataSingle() = default;
    void setData(py::array_t<float> coords, py::list labels);
    py::array_t<float> getCoordinates() const;
    // void addPoint(float x, float y, const std::string &label);
    // void addPoints(const py::array_t<float> &coordinates, const py::list &labels);
    void buildKDTree();

    py::list radiusSearch(py::array_t<float> &cellPos, float radius = 100.0);

    py::list buildFPTreeKNN(const py::object &cell_pos,
                             int k = 30,
                             float minSupport = 0.5,
                             bool disDuplicates = false,
                             bool ifMax = true,
                             float maxDist = 200.0);

    std::set<DuplicatePattern> buildFPTreeKNN(const Positions &cell_pos,
                                              int k = 30,
                                              float minSupport = 0.5,
                                              bool disDuplicates = false,
                                              bool ifMax = true,
                                              float maxDist = 200.0);

    py::tuple buildFPTreeDist(const py::object &cell_pos,
                              float radius = 100.0,
                              float minSupport = 0.5,
                              bool disDuplicates = false,
                              bool ifMax = true,
                              int minSize = 0);

    std::tuple<std::set<Pattern>, std::vector<std::map<Item, int>>, Idxs> buildFPTreeDist(
        const std::vector<std::vector<float>> &cellPos,
        float radius = 100.0,
        float minSupport = 0.5,
        bool disDuplicates = false,
        bool ifMax = true,
        int minSize = 0);

    py::list motifEnrichmentKNN(const std::string &ct,
                                const py::list &motifs,
                                int k = 30,
                                float minSupport = 0.5,
                                bool disDuplicates = false,
                                float maxDist = 200.0);

    py::list motifEnrichmentDist(const std::string &ct,
                                 const py::list &motifs,
                                 float radius = 100.0,
                                 float minSupport = 0.5,
                                 bool disDuplicates = false,
                                 int minSize = 0);
};
