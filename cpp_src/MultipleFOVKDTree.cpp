#include "MultipleFOVKDTree.h"
#include "utils.h"
#include "FPGrowth.h"

void SpatialDataMultiple::setFOVData(int fovID, const py::array_t<float> &coordinates, const py::list &labels)
{
    SpatialDataSingle &fovData = fovs[fovID]; // If fovID doesn't exist, value-intialize fovs[fovID] and set values for this SpatialDataSingle object
    fovData.setData(coordinates, labels);
    fovData.buildKDTree();
}

py::list SpatialDataMultiple::buildFPTreeKNN(const Item &cellType, const py::list &fovIDs, int k, float minSupport, bool disDuplicates)
{
    std::vector<Transaction> transactions;

    for (auto fovID : fovIDs)
    {
        int fov = fovID.cast<int>();

        const SpatialDataSingle &singleFOV = fovs[fov];
        const std::vector<Item> &labels = singleFOV.labels;

        // Find the indices of cellType
        std::vector<size_t> cinds;
        for (size_t i = 0; i < labels.size(); ++i)
        {
            if (labels[i] == cellType)
                cinds.push_back(i);
        }
        if (cinds.size() == 0)
        {
            continue;
        }

        transactions.reserve(transactions.size() + cinds.size());

        // Populate the spatial coordinates of cellType
        auto r = singleFOV.coordinates.template unchecked<2>();
        py::array_t<float> ctPosFOV;
        auto ctPosFOVPtr = ctPosFOV.mutable_unchecked<2>();

        for (size_t i = 0; i < cinds.size(); ++i)
        {
            ctPosFOVPtr(i, 0) = r(cinds[i], 0);
            ctPosFOVPtr(i, 1) = r(cinds[i], 1);
        }

        // Find the kNNs for cellType
        for (ssize_t i = 0; i < ctPosFOV.shape(0); i++)
        {
            pcl::PointXY searchPoint{ctPosFOVPtr(i, 0), ctPosFOVPtr(i, 1)};
            std::vector<int> pointIdxNKNSearch(k + 1);
            std::vector<float> pointNKNSquareDistance(k + 1);
            singleFOV.kdtree.nearestKSearch(searchPoint, k + 1, pointIdxNKNSearch, pointNKNSquareDistance);

            Transaction transaction;
            transaction.reserve(k + 1);
            for (size_t j = 1; j < pointIdxNKNSearch.size(); j++)
            {
                transaction.push_back(labels[pointIdxNKNSearch[j]]);
            }

            if (disDuplicates)
            {
                transaction = distinguishDuplicates(transaction);
            }
            if (transaction.size() > 0)
            {
                transactions.push_back(transaction);
            }
        }
    }

    FPTree fpTree(transactions, static_cast<uint64_t>(std::ceil(minSupport * transactions.size())));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);

    frequentPatterns = findMaximalPatterns(frequentPatterns);

    std::set<DuplicatePattern> frequentDuplicatePatterns;
    if (disDuplicates)
    {
        frequentDuplicatePatterns = removeSuffix(frequentPatterns);
    }
    else
    {
        for (const auto &pattern : frequentPatterns)
        {
            DuplicateItemset itemset(pattern.first.begin(), pattern.first.end());
            DuplicatePattern duplicatePattern(itemset, pattern.second);
            frequentDuplicatePatterns.insert(duplicatePattern);
        }
    }

    py::list result;
    for (const auto &pattern : frequentDuplicatePatterns)
    {
        py::dict dict;
        py::list items;
        for (const auto &item : pattern.first)
        {
            items.append(py::cast(item));
        }
        dict["items"] = items;
        dict["support"] = py::cast(static_cast<double>(pattern.second) / transactions.size());
        result.append(dict);
    }

    return result;
}

py::list SpatialDataMultiple::buildFPTreeDist(const Item &cellType, const py::list &fovIDs, float radius, float minSupport, bool disDuplicates, int minSize)
{
    std::vector<Transaction> transactions;

    for (auto fovID : fovIDs)
    {
        int fov = fovID.cast<int>();
        const SpatialDataSingle &singleFOV = fovs[fov];
        const std::vector<Item> &labels = singleFOV.labels;

        // Find the indices of cellType
        std::vector<size_t> cinds;
        for (size_t i = 0; i < labels.size(); ++i)
        {
            if (labels[i] == cellType)
            {
                cinds.push_back(i);
            }
        }
        if (cinds.size() == 0)
        {
            continue;
        }

        transactions.reserve(transactions.size() + cinds.size());

        // Populate the spatial coordinates of cellType
        auto r = singleFOV.coordinates.template unchecked<2>();
        py::array_t<float> ctPosFOV;
        auto ctPosFOVPtr = ctPosFOV.mutable_unchecked<2>();

        for (size_t i = 0; i < cinds.size(); ++i)
        {
            ctPosFOVPtr(i, 0) = r(cinds[i], 0);
            ctPosFOVPtr(i, 1) = r(cinds[i], 1);
        }

        // Find the neighbors within maxDist
        for (ssize_t i = 0; i < ctPosFOV.shape(0); ++i)
        {
            pcl::PointXY searchPoint{ctPosFOVPtr(i, 0), ctPosFOVPtr(i, 1)};
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquareDistance;

            singleFOV.kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance);

            Transaction transaction;
            transaction.reserve(pointIdxRadiusSearch.size());
            if (pointIdxRadiusSearch.size() > static_cast<size_t>(minSize))
            {
                for (size_t j = 1; j < pointIdxRadiusSearch.size(); ++j)
                {
                    transaction.push_back(labels[pointIdxRadiusSearch[j]]);
                    ;
                }

                if (disDuplicates)
                {
                    transaction = distinguishDuplicates(transaction);
                }
                if (transaction.size() > 0)
                {
                    transactions.push_back(transaction);
                }
            }
        }
    }
    FPTree fpTree(transactions, static_cast<uint64_t>(std::ceil(minSupport * transactions.size())));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);

    frequentPatterns = findMaximalPatterns(frequentPatterns);
    std::set<DuplicatePattern> frequentDuplicatePatterns;
    if (disDuplicates)
    {
        frequentDuplicatePatterns = removeSuffix(frequentPatterns);
    }
    else
    {
        for (const auto &pattern : frequentPatterns)
        {
            DuplicateItemset itemset(pattern.first.begin(), pattern.first.end());
            DuplicatePattern duplicatePattern(itemset, pattern.second);
            frequentDuplicatePatterns.insert(duplicatePattern);
        }
    }

    py::list result;
    for (const auto &pattern : frequentDuplicatePatterns)
    {
        py::dict dict;
        py::list items;
        for (const auto &item : pattern.first)
        {
            items.append(py::cast(item));
        }
        dict["items"] = items;
        dict["support"] = py::cast(static_cast<double>(pattern.second) / transactions.size());
        result.append(dict);
    }

    return result;
}

py::list SpatialDataMultiple::motifEnrichmentKNN(const Item &cellType, const py::list &motifs, py::list &fovIDs, int k, float minSupport, bool disDuplicates, float maxDist)
{
    py::list motifsOut;
    for (const auto &motif : motifs)
    {
        int nLabels = 0;
        int nCt = 0;
        int nMotifLabels = 0;
        int nMotifCt = 0;

        // Get motifs
        std::vector<Item> motifVec;
        for (const auto &item : motif)
        {
            motifVec.push_back(item.cast<Item>());
        }
        std::vector<Item> sortMotif = motifVec;
        std::sort(sortMotif.begin(), sortMotif.end()); // Sort items in motif

        for (auto fovID : fovIDs)
        {
            int fov = fovID.cast<int>();

            const SpatialDataSingle &singleFOV = fovs[fov];
            const std::vector<Item> &labels = singleFOV.labels;

            Idxs idxs(singleFOV.coordinates.shape(0));
            std::vector<std::vector<float>> dists(singleFOV.coordinates.shape(0));
            auto r = singleFOV.coordinates.template unchecked<2>();
            for (ssize_t i = 0; i < singleFOV.coordinates.shape(0); i++)
            {
                std::vector<int> pointIdxs(k + 1);
                std::vector<float> pointDists(k + 1);
                pcl::PointXY searchPoint{r(i, 0), r(i, 1)};
                singleFOV.kdtree.nearestKSearch(searchPoint, k + 1, pointIdxs, pointDists);
                idxs[i] = std::move(pointIdxs);
                dists[i] = std::move(pointDists);
            }

            std::vector<int> cellTypeIndxs;
            Positions cellTypePos;
            for (size_t i = 0; i < labels.size(); i++)
            {
                if (labels[i] == cellType)
                {
                    cellTypeIndxs.push_back(i);
                    cellTypePos.push_back({r(i, 0), r(i, 1)});
                }
            }

            std::pair<int, int> result = motifCalculation(sortMotif, idxs, dists, labels, cellTypeIndxs, maxDist);

            int nMotifCtFOV = result.first;
            int nMotifLabelsFOV = result.second;
            int nCtFOV = cellTypeIndxs.size();
            int nLabelsFOV = labels.size();

            nCt += nCtFOV;
            nLabels += nLabelsFOV;
            nMotifCt += nMotifCtFOV;
            nMotifLabels += nMotifLabelsFOV;
        }
        int count = std::count(sortMotif.begin(), sortMotif.end(), cellType);
        if (count > 0)
        {
            nCt = std::round(nCt / count);
        }

        py::dict motifOut;
        motifOut["center"] = cellType;
        motifOut["motifs"] = sortMotif;
        motifOut["n_center_motif"] = nMotifCt;
        motifOut["n_center"] = nCt;
        motifOut["n_motif"] = nMotifLabels;
        motifOut["n_labels"] = nLabels;
        motifOut["motifs"] = motif;
        motifsOut.append(motifOut);
    }
    return motifsOut;
}

py::list SpatialDataMultiple::motifEnrichmentDist(const Item &cellType, const py::list &motifs, py::list &fovIDs, float radius, float minSupport, bool disDuplicates, int minSize)
{
    py::list motifsOut;
    for (const auto &motif : motifs)
    {
        int nLabels = 0;
        int nCt = 0;
        int nMotifLabels = 0;
        int nMotifCt = 0;

        // Get motifs
        std::vector<Item> motifVec;
        for (const auto &item : motif)
        {
            motifVec.push_back(item.cast<Item>());
        }
        auto sortMotif = motifVec;
        std::sort(sortMotif.begin(), sortMotif.end()); // Sort items in motif

        for (auto fovID : fovIDs)
        {
            int fov = fovID.cast<int>();

            const SpatialDataSingle &singleFOV = fovs[fov];
            const std::vector<Item> &labels = singleFOV.labels;

            Idxs idxs(singleFOV.coordinates.shape(0));
            std::vector<std::vector<float>> dists(singleFOV.coordinates.shape(0));
            auto r = singleFOV.coordinates.template unchecked<2>();
            for (ssize_t i = 0; i < singleFOV.coordinates.shape(0); i++)
            {
                std::vector<int> pointIdxs;
                std::vector<float> pointDists;
                pcl::PointXY searchPoint{r(i, 0), r(i, 1)};
                singleFOV.kdtree.radiusSearch(searchPoint, radius, pointIdxs, pointDists);
                idxs[i] = std::move(pointIdxs);
                dists[i] = std::move(pointDists);
            }

            std::vector<int> cellTypeIndxs;
            Positions cellTypePos;
            for (size_t i = 0; i < labels.size(); i++)
            {
                if (labels[i] == cellType)
                {
                    cellTypeIndxs.push_back(i);
                    cellTypePos.push_back({r(i, 0), r(i, 1)});
                }
            }

            auto result = motifCalculation(sortMotif, idxs, dists, labels, cellTypeIndxs, 100000000);

            int nMotifCtFOV = result.first;
            int nMotifLabelsFOV = result.second;
            int nCtFOV = cellTypeIndxs.size();
            int nLabelsFOV = labels.size();

            nCt += nCtFOV;
            nLabels += nLabelsFOV;
            nMotifCt += nMotifCtFOV;
            nMotifLabels += nMotifLabelsFOV;
        }
        int count = std::count(sortMotif.begin(), sortMotif.end(), cellType);
        if (count > 0)
        {
            nCt = std::round(nCt / count);
        }

        py::dict motifOut;
        motifOut["center"] = cellType;
        motifOut["motifs"] = sortMotif;
        motifOut["n_center_motif"] = nMotifCt;
        motifOut["n_center"] = nCt;
        motifOut["n_motif"] = nMotifLabels;
        motifOut["n_labels"] = nLabels;
        motifOut["motifs"] = motif;
        motifsOut.append(motifOut);
    }
    return motifsOut;
}

py::list SpatialDataMultiple::findFPKnnFOV(const std::string &cellType, const int fovID, int k,
                                           float minSupport)
{
    SpatialDataSingle &singleFOV = fovs[fovID];
    const std::vector<Item> &labels = singleFOV.labels;
    std::vector<size_t> cinds;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        if (labels[i] == cellType)
            cinds.push_back(i);
    }
    auto r = singleFOV.coordinates.template unchecked<2>();
    py::array_t<float> ctPosFOV;
    auto ctPosFOVPtr = ctPosFOV.mutable_unchecked<2>();
    for (size_t i = 0; i < cinds.size(); ++i)
    {
        ctPosFOVPtr(i, 0) = r(cinds[i], 0);
        ctPosFOVPtr(i, 1) = r(cinds[i], 1);
    }
    py::list fp = singleFOV.buildFPTreeKNN(ctPosFOV, k, minSupport, false, false, 500);
    return fp;
}

py::list SpatialDataMultiple::findFPDistFOV(const std::string &cellType, const int fovID, float maxDist,
                                            int minSize, float minSupport)
{
    SpatialDataSingle &singleFOV = fovs[fovID];
    const std::vector<Item> &labels = singleFOV.labels;
    std::vector<size_t> cinds;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        if (labels[i] == cellType)
            cinds.push_back(i);
    }

    auto r = singleFOV.coordinates.template unchecked<2>();
    py::array_t<float> ctPosFOV;
    auto ctPosFOVPtr = ctPosFOV.mutable_unchecked<2>();
    for (size_t i = 0; i < cinds.size(); ++i)
    {
        ctPosFOVPtr(i, 0) = r(cinds[i], 0);
        ctPosFOVPtr(i, 1) = r(cinds[i], 1);
    }

    py::tuple fpOut = singleFOV.buildFPTreeDist(ctPosFOV, maxDist, minSupport, false, false, minSize);
    py::list fp = fpOut[0].cast<py::list>();

    return fp;
}

py::tuple SpatialDataMultiple::differentialAnalysisKnn(const std::string &cellType,
                                                       const py::list &datasets,
                                                       const py::list &fovsID1,
                                                       const py::list &fovsID2,
                                                       int k,
                                                       float minSupport)
{
    // Identify frequent patterns in each dataset
    std::vector<std::unordered_map<std::string, std::vector<double>>> fpDatasets(2);
    std::vector<std::unordered_set<std::string>> itemsets(2);

    for (size_t i = 0; i < 2; ++i)
    {
        const py::list &fovsID = (i == 0) ? fovsID1 : fovsID2;
        std::string datasetName = datasets[i].cast<std::string>();

        // std::unordered_map<std::string, std::unordered_map<std::string, double>> patternSupport;
        for (const auto &fovID : fovsID)
        {
            int fov = py::cast<int>(fovID);

            py::list fpFov = findFPKnnFOV(cellType, fov, k, minSupport);
            if (fpFov.empty())
            {
                continue;
            }

            for (size_t j = 0; j < fpFov.size(); ++j)
            {
                // Build the string representation of itemset for indexing support values
                py::list itemlist = fpFov[j]["items"];
                std::ostringstream oss;
                bool first = true;
                for (const auto &item : itemlist)
                {
                    if (!first)
                    {
                        oss << ", ";
                    }
                    oss << item.cast<std::string>();
                    first = false;
                }
                std::string itemset = oss.str();
                double support = fpFov[j]["support"].cast<double>();
                fpDatasets[i][itemset].push_back(support);
                itemsets[i].insert(itemset);
            }
        }
    }
    py::dict result1, result2;
    for (const auto &itemset : itemsets[0])
    {
        result1[py::cast(itemset)] = py::cast(fpDatasets[0]);
    }
    for (const auto &itemset : itemsets[1])
    {
        result2[py::cast(itemset)] = py::cast(fpDatasets[1]);
    }

    return py::make_tuple(result1, result2);
}

py::tuple SpatialDataMultiple::differentialAnalysisDist(const std::string &cellType,
                                                        const py::list &datasets,
                                                        const py::list &fovsID1,
                                                        const py::list &fovsID2,
                                                        float maxDist,
                                                        float minSupport,
                                                        int minSize)
{
    // Identify frequent patterns in each dataset
    std::vector<std::unordered_map<std::string, std::vector<double>>> fpDatasets(2);
    std::vector<std::unordered_set<std::string>> itemsets(2);

    for (size_t i = 0; i < 2; ++i)
    {
        const py::list &fovsID = (i == 0) ? fovsID1 : fovsID2;
        std::string datasetName = datasets[i].cast<std::string>();

        // std::unordered_map<std::string, std::unordered_map<std::string, double>> patternSupport;
        for (const auto &fovID : fovsID)
        {
            int fov = py::cast<int>(fovID);

            py::list fpFov = findFPDistFOV(cellType, fov, maxDist, minSize, minSupport);
            if (fpFov.empty())
            {
                continue;
            }

            for (size_t j = 0; j < fpFov.size(); ++j)
            {
                // Build the string representation of itemset for indexing support values
                py::list itemlist = fpFov[j]["items"];
                std::ostringstream oss;
                bool first = true;
                for (const auto &item : itemlist)
                {
                    if (!first)
                    {
                        oss << ", ";
                    }
                    oss << item.cast<std::string>();
                    first = false;
                }
                std::string itemset = oss.str();
                double support = fpFov[j]["support"].cast<double>();
                fpDatasets[i][itemset].push_back(support);
                itemsets[i].insert(itemset);
            }
        }
    }
    py::dict result1, result2;
    for (const auto &itemset : itemsets[0])
    {
        result1[py::cast(itemset)] = py::cast(fpDatasets[0]);
    }
    for (const auto &itemset : itemsets[1])
    {
        result2[py::cast(itemset)] = py::cast(fpDatasets[1]);
    }

    return py::make_tuple(result1, result2);
}

PYBIND11_MODULE(spatial_module, m)
{
    py::class_<SpatialDataSingle>(m, "SpatialDataSingle")
        .def(py::init<>())
        .def("set_data", &SpatialDataSingle::setData, "Set the spatial coordinates and labels")
        .def("get_coordinates", &SpatialDataSingle::getCoordinates, "Get the spatial coordinates")
        // .def("add_point", &SpatialDataSingle::addPoint, "Add a single point")
        // .def("add_points", &SpatialDataSingle::addPoints, "Add multiple points")
        .def("build_kdtree", &SpatialDataSingle::buildKDTree, "Build the KD-tree")
        .def("radius_search", &SpatialDataSingle::radiusSearch,
             py::arg("cell_pos"),
             py::arg("radius") = 100.0,
             "Get radius search indices by kdtree")
        .def("build_fptree_knn", py::overload_cast<const py::object &, int, float, bool, bool, float>(&SpatialDataSingle::buildFPTreeKNN),
             py::arg("cell_pos"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("if_max") = true,
             py::arg("max_dist") = 200.0,
             "Build FP-tree using KNN")
        .def("build_fptree_dist", py::overload_cast<const py::object &, float, float, bool, bool, int>(&SpatialDataSingle::buildFPTreeDist),
             py::arg("cell_pos"),
             py::arg("radius") = 100.0,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("if_max") = true,
             py::arg("min_size") = 0,
             "Build FP-tree using distance")
        .def("motif_enrichment_knn", &SpatialDataSingle::motifEnrichmentKNN,
             py::arg("ct"),
             py::arg("motifs"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("max_dist") = 200.0,
             "Perform motif enrichment using KNN")
        .def("motif_enrichment_dist", &SpatialDataSingle::motifEnrichmentDist,
             py::arg("ct"),
             py::arg("motifs"),
             py::arg("radius") = 100.0,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("min_size") = 0,
             "Perform motif enrichment using distance");

    py::class_<SpatialDataMultiple>(m, "SpatialDataMultiple")
        .def(py::init<>())
        .def("set_fov_data", &SpatialDataMultiple::setFOVData, "Build multiple FOVs")
        .def("find_fp_knn", &SpatialDataMultiple::buildFPTreeKNN,
             py::arg("cell_type"),
             py::arg("fov_ids"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             "Identify frequent patterns in specified FOVs using KNN")
        .def("find_fp_dist", &SpatialDataMultiple::buildFPTreeDist,
             py::arg("cell_type"),
             py::arg("fov_ids"),
             py::arg("radius") = 100.0,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("min_size") = 0,
             "Identify frequent patterns in specified FOVs using radius-based neighborhood")
        .def("motif_enrichment_knn", &SpatialDataMultiple::motifEnrichmentKNN,
             py::arg("cell_type"),
             py::arg("motifs"),
             py::arg("fov_ids"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("max_dist") = 200.0,
             "Perform motif enrichment using KNN")
        .def("motif_enrichment_dist", &SpatialDataMultiple::motifEnrichmentDist,
             py::arg("cell_type"),
             py::arg("motifs"),
             py::arg("fov_ids"),
             py::arg("radius") = 100.0,
             py::arg("min_support") = 0.5,
             py::arg("dis_duplicates") = false,
             py::arg("min_size") = 0,
             "Perform motif enrichment using radius-based neighborhood")
        .def("find_fp_knn_fov", &SpatialDataMultiple::findFPKnnFOV,
             py::arg("cell_type"),
             py::arg("fov_id"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             "Find frequent patterns in each single FOV with KNN")
        .def("find_fp_dist_fov", &SpatialDataMultiple::findFPDistFOV,
             py::arg("cell_type"),
             py::arg("fov_id"),
             py::arg("radius") = 100.0,
             py::arg("min_size") = 0,
             py::arg("min_support") = 0.5,
             "Find frequent patterns in each single FOV with radius-based neighborhood")
        .def("differential_analysis_knn", &SpatialDataMultiple::differentialAnalysisKnn,
             py::arg("cell_type"),
             py::arg("datasets"),
             py::arg("fovs_id0"),
             py::arg("fovs_id1"),
             py::arg("k") = 30,
             py::arg("min_support") = 0.5,
             "Perform differential analysis of frequent patterns using KNN")
        .def("differential_analysis_dist", &SpatialDataMultiple::differentialAnalysisDist,
             py::arg("cell_type"),
             py::arg("datasets"),
             py::arg("fovs_id0"),
             py::arg("fovs_id1"),
             py::arg("radius") = 100.0,
             py::arg("min_support") = 0.5,
             py::arg("min_size") = 0,
             "Perform differential analysis of frequent patterns using radius-based neighborhood");
}