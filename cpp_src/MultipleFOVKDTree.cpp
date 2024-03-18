#include "MultipleFOVKDTree.h"

void SpatialDataMultiple::addFOVData(int fovID, const py::array_t<float> &coordinates, const std::vector<std::string> &labels){
    SpatialDataSingle& fovData=fovs[fovID]; // If fovID doesn't exist, value-intialize fovs[fovID] and set values for this SpatialDataSingle object
    fovData.addPoints(coordinates, labels);
    fovData.buildKDTree();
}




PYBIND11_MODULE(spatial_module, m) {
    py::class_<SpatialDataMultiple>(m, "SpatialDataMultiple")
        .def(py::init<>())
        .def("addFOVData", &SpatialDataMultiple::addFOVData);
}