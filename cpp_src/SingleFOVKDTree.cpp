#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
// #include <chrono>

#include "SingleFOVKDTree.h"
#include "FPGrowth.h"
#include "utils.h"

void SpatialDataSingle::setData(py::array_t<float> coords, py::list labels)
{
    this->coordinates = std::move(coords);
    std::vector<std::string> tempLabels;
    for (const auto &item : labels)
    {
        tempLabels.push_back(item.cast<std::string>());
    }
    this->labels = std::move(tempLabels);
}

py::array_t<float> SpatialDataSingle::getCoordinates() const
{
    return this->coordinates;
}
// void SpatialDataSingle::addPoint(float x, float y, const std::string &label)
// {
//     points->push_back(pcl::PointXY(x, y));
//     labels.push_back(label);
// }

// void SpatialDataSingle::addPoints(const py::array_t<float> &coordinates, const py::list &labels)
// {
//     auto r = coordinates.uncheck<2>(); // Ensureit's of shape (N, 2) for N points
//     for (ssize_t i = 0; i < r.shape(0); i++)
//     {
//         float x = r(i, 0);
//         float y = r(i, 1);
//         std::string label = labels[i].cast<std::string>();
//         this->addPoint(x, y, label);
//     }
// }

void SpatialDataSingle::buildKDTree()
{
    auto coords = this->coordinates.template unchecked<2>();
    auto numPoints = coords.shape(0);
    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    cloud->resize(numPoints);
    for (py::ssize_t i = 0; i < numPoints; i++)
    {
        (*cloud)[i].x = coords(i, 0);
        (*cloud)[i].y = coords(i, 1);
    }
    kdtree.setInputCloud(cloud);
}

py::list SpatialDataSingle::radiusSearch(py::array_t<float> &cellPos, float radius)
{
    auto r = cellPos.unchecked<2>();
    py::list index;

    for (ssize_t i = 0; i < r.shape(0); i++)
    {
        pcl::PointXY searchPoint{r(i, 0), r(i, 1)};
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquareDistance;

        kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance);

        py::list indices;
        for (const auto &idx : pointIdxRadiusSearch)
        {
            indices.append(idx);
        }
        index.append(indices);
    }
    return index;
}

py::list SpatialDataSingle::buildFPTreeKNN(const py::object &cellPos,
                                           int k,
                                           float minSupport,
                                           bool disDuplicates,
                                           bool ifMax,
                                           float maxDist)
{
    py::array_t<float> cellPosArray;
    if (cellPos.is_none())
    {
        cellPosArray = this->coordinates;
    }
    else
    {
        cellPosArray = cellPos.cast<py::array_t<float>>();
    }
    auto r = cellPosArray.template unchecked<2>(); // Ensure cellPos is of shape (N, 2) and access numpy array read-only
    std::vector<Transaction> transactions;         // Prepare transactions data structure
    transactions.reserve(r.shape(0));
    float maxDistSquare = maxDist * maxDist;

    // For each point in cellPos, perform k-NN search and filter results
    // auto start = std::chrono::high_resolution_clock::now();
    for (ssize_t i = 0; i < r.shape(0); i++)
    {
        // Perform k-NN search with PCL
        std::vector<int> pointIdxNKNSearch(k + 1);
        std::vector<float> pointNKNSquaredDistance(k + 1);
        pcl::PointXY searchPoint{r(i, 0), r(i, 1)};
        kdtree.nearestKSearch(searchPoint, k + 1, pointIdxNKNSearch, pointNKNSquaredDistance);

        // Filter neighbors based on maxDistSquare and construct a transaction from the results
        Transaction transaction;
        transaction.reserve(k + 1);

        for (size_t j = 1; j < pointIdxNKNSearch.size(); j++)
        {
            if (pointNKNSquaredDistance[j] <= maxDistSquare)
            {
                // Add the label of the neighbor to the transaction, do not include center cell itself
                transaction.push_back(this->labels[pointIdxNKNSearch[j]]);
            }
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
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Running time to build transactions: " << duration.count() << " microseconds" << std::endl;

    // Construct the FP-tree from transactions and mine it for patterns
    // start = std::chrono::high_resolution_clock::now();
    FPTree fpTree(transactions, static_cast<uint64_t>(std::ceil(minSupport * transactions.size())));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Running time for frequentPatterns: " << duration.count() << " microseconds" << std::endl;
    // std::cout << "number of patterns: " << frequentPatterns.size() << std::endl;

    if (ifMax)
    {
        // start = std::chrono::high_resolution_clock::now();
        frequentPatterns = findMaximalPatterns(frequentPatterns);
        // end = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Running time for find max: " << duration.count() << " microseconds" << std::endl;
    }
    std::set<DuplicatePattern> frequentDuplicatePatterns;

    if (disDuplicates)
    {
        // start = std::chrono::high_resolution_clock::now();
        frequentDuplicatePatterns = removeSuffix(frequentPatterns);
        // end = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Running time for remove suffix: " << duration.count() << " microseconds" << std::endl;
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

std::set<DuplicatePattern> SpatialDataSingle::buildFPTreeKNN(const Positions &cellPos,
                                                             int k,
                                                             float minSupport,
                                                             bool disDuplicates,
                                                             bool ifMax,
                                                             float maxDist)
{
    std::vector<Transaction> transactions; // Prepare transactions data structure
    transactions.reserve(cellPos.size());
    float maxDistSquare = maxDist * maxDist;

    // For each point in cellPos, perform k-NN search and filter results
    for (size_t i = 0; i < cellPos.size(); i++)
    {
        // Perform k-NN search with PCL
        std::vector<int> pointIdxNKNSearch(k + 1);
        std::vector<float> pointNKNSquaredDistance(k + 1);
        pcl::PointXY searchPoint{cellPos[i][0], cellPos[i][1]};
        kdtree.nearestKSearch(searchPoint, k + 1, pointIdxNKNSearch, pointNKNSquaredDistance);

        // Filter neighbors based on maxDistSquare and construct a transaction from the results
        Transaction transaction;
        transaction.reserve(k + 1);

        for (size_t j = 1; j < pointIdxNKNSearch.size(); j++)
        {
            if (pointNKNSquaredDistance[j] <= maxDistSquare)
            {
                // Add the label of the neighbor to the transaction, do not include center cell itself
                transaction.push_back(this->labels[pointIdxNKNSearch[j]]);
            }
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
    // Construct the FP-tree from transactions and mine it for patterns
    FPTree fpTree(transactions, static_cast<uint64_t>(minSupport * transactions.size()));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);

    if (ifMax)
    {
        frequentPatterns = findMaximalPatterns(frequentPatterns);
    }
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

    return frequentDuplicatePatterns;
}

py::tuple SpatialDataSingle::buildFPTreeDist(
    const py::object &cellPos,
    float radius,
    float minSupport,
    bool disDuplicates,
    bool ifMax,
    int minSize)
{
    py::array_t<float> cellPosArray;
    if (cellPos.is_none())
    {
        cellPosArray = this->coordinates;
    }
    else
    {
        cellPosArray = cellPos.cast<py::array_t<float>>();
    }
    auto r = cellPosArray.template unchecked<2>();

    std::vector<Transaction> transactions;
    transactions.reserve(r.shape(0));
    std::vector<std::map<Item, int>> cellTypeOccurrences;
    Idxs validIdx;

    // For each point in cellPos, perform radius-based search and filter results
    // auto start = std::chrono::high_resolution_clock::now();
    for (ssize_t i = 0; i < r.shape(0); i++)
    {
        // Perform radius-based search with PCL
        pcl::PointXY searchPoint{r(i, 0), r(i, 1)};
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquareDistance;

        kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance);

        // Construct transactions
        Transaction transaction;
        if (pointIdxRadiusSearch.size() > static_cast<size_t>(minSize))
        {
            std::map<Item, int> cellTypeCounts;
            for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)
            {
                ++cellTypeCounts[this->labels[pointIdxRadiusSearch[j]]];
                transaction.push_back(this->labels[pointIdxRadiusSearch[j]]);
            }
            if (disDuplicates)
            {
                transaction = distinguishDuplicates(transaction);
            }
            if (transaction.size() > 0)
            {
                transactions.push_back(transaction);
                validIdx.push_back(pointIdxRadiusSearch);
                cellTypeOccurrences.push_back(cellTypeCounts);
            }
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Running time to build transactions: " << duration.count() << " microseconds" << std::endl;

    // Construct the FP-tree from transactions and mine it for patterns
    // start = std::chrono::high_resolution_clock::now();
    FPTree fpTree(transactions, static_cast<uint64_t>(minSupport * transactions.size()));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Running time for frequentPatterns: " << duration.count() << " microseconds" << std::endl;
    // std::cout << "number of patterns: " << frequentPatterns.size() << std::endl;

    if (ifMax)
    {
        // start = std::chrono::high_resolution_clock::now();
        frequentPatterns = findMaximalPatterns(frequentPatterns);
        // end = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Running time for find max: " << duration.count() << " microseconds" << std::endl;
    }

    std::set<DuplicatePattern> frequentDuplicatePatterns;
    if (disDuplicates)
    {
        // start = std::chrono::high_resolution_clock::now();
        frequentDuplicatePatterns = removeSuffix(frequentPatterns);
        // end = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Running time for remove suffix: " << duration.count() << " microseconds" << std::endl;
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

    // Convert cpp objects into py objects
    py::list patternList;
    for (const auto &pattern : frequentDuplicatePatterns)
    {
        py::dict patternDict;
        py::list items;
        for (const auto &item : pattern.first)
        {
            items.append(py::cast(item));
        }
        patternDict["items"] = items;
        patternDict["support"] = py::cast(static_cast<double>(pattern.second) / transactions.size());
        patternList.append(patternDict);
    }

    py::list cellTypeOccList;
    for (const auto &occurrence : cellTypeOccurrences)
    {
        py::dict occurrenceDict;
        for (const auto &pair : occurrence)
        {
            occurrenceDict[py::cast(pair.first)] = py::cast(pair.second);
        }
        cellTypeOccList.append(occurrenceDict);
    }

    py::list validIdxList;
    for (const auto &indices : validIdx)
    {
        validIdxList.append(py::cast(indices));
    }

    return py::make_tuple(patternList, cellTypeOccList, validIdxList);
}

std::tuple<std::set<Pattern>, std::vector<std::map<Item, int>>, Idxs> SpatialDataSingle::buildFPTreeDist(
    const std::vector<std::vector<float>> &cellPos,
    float radius,
    float minSupport,
    bool disDuplicates,
    bool ifMax,
    int minSize)
{
    std::vector<Transaction> transactions;
    transactions.reserve(cellPos.size());
    std::vector<std::map<Item, int>> cellTypeOccurrences;
    Idxs validIdx;

    // For each point in cellPos, perform radius-based search and filter results
    for (size_t i = 0; i < cellPos.size(); i++)
    {
        // Perform radius-based search with PCL
        pcl::PointXY searchPoint{cellPos[i][0], cellPos[i][1]};
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquareDistance;

        kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance);

        // Construct transactions
        Transaction transaction;
        if (pointIdxRadiusSearch.size() > static_cast<size_t>(minSize))
        {
            std::map<Item, int> cellTypeCounts;
            for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)
            {
                ++cellTypeCounts[this->labels[pointIdxRadiusSearch[j]]];
                transaction.push_back(this->labels[pointIdxRadiusSearch[j]]);
            }
            if (disDuplicates)
            {
                transaction = distinguishDuplicates(transaction);
            }
            if (transaction.size() > 0)
            {
                transactions.push_back(transaction);
                validIdx.push_back(pointIdxRadiusSearch);
                cellTypeOccurrences.push_back(cellTypeCounts);
            }
        }
    }

    // Construct the FP-tree from transactions and mine it for patterns
    FPTree fpTree(transactions, static_cast<uint64_t>(minSupport * transactions.size()));
    std::set<Pattern> frequentPatterns = fptree_growth(fpTree);

    if (ifMax)
    {
        frequentPatterns = findMaximalPatterns(frequentPatterns);
    }
    std::set<DuplicatePattern> frequentDuplicatePatterns;
    if (disDuplicates)
    {
        frequentDuplicatePatterns = removeSuffix(frequentPatterns);
    }
    return std::make_tuple(frequentPatterns, cellTypeOccurrences, validIdx);
}

py::list SpatialDataSingle::motifEnrichmentKNN(
    const std::string &ct,
    const py::list &motifs,
    int k,
    float minSupport,
    bool disDuplicates,
    float maxDist)
{
    Idxs idxs(this->coordinates.shape(0));
    std::vector<std::vector<float>> dists(this->coordinates.shape(0));
    // Use k+1 to find the kNNs of all cells except for the points themselves
    auto coords = this->coordinates.template unchecked<2>();
    for (ssize_t i = 0; i < this->coordinates.shape(0); i++)
    {
        std::vector<int> pointIdxs(k + 1);
        std::vector<float> pointDists(k + 1);
        pcl::PointXY searchPoint{coords(i, 0), coords(i, 1)};
        kdtree.nearestKSearch(searchPoint, k + 1, pointIdxs, pointDists);
        idxs[i] = std::move(pointIdxs);
        dists[i] = std::move(pointDists);
    }

    std::vector<int> cellTypeIndxs;
    Positions cellTypePos;
    for (size_t i = 0; i < this->labels.size(); i++)
    {
        if (this->labels[i] == ct)
        {
            cellTypeIndxs.push_back(i);
            cellTypePos.push_back({coords(i, 0), coords(i, 1)});
        }
    }

    // Get motifs
    py::list motifsOut;
    py::list motifsToUse;
    if (motifs.size() == 0)
    {
        // Replace with the appropriate frequent pattern mining function
        std::set<DuplicatePattern> frequentPatterns = buildFPTreeKNN(cellTypePos, k, minSupport, disDuplicates);

        for (const auto &pattern : frequentPatterns)
        {
            py::list itemset;
            for (const auto &item : pattern.first)
            {
                itemset.append(item);
            }
            motifsToUse.append(itemset);
        }
    }
    else
    {
        motifsToUse = motifs;
    }

    if (motifsToUse.size() == 0)
    {
        throw std::runtime_error("No frequent patterns were found. Please lower support threshold.");
    }

    for (const auto &motif : motifsToUse)
    {
        std::vector<Item> motifVec;
        for (const auto &item : motif)
        {
            motifVec.push_back(item.cast<Item>());
        }

        auto sortMotif = motifVec;
        std::sort(sortMotif.begin(), sortMotif.end()); // Sort items in motif

        auto result = motifCalculation(sortMotif, idxs, dists, this->labels, cellTypeIndxs, maxDist);
        int nMotifCt = result.first;
        int nMotifLabels = result.second;

        int nCt = cellTypeIndxs.size();
        int count = std::count(sortMotif.begin(), sortMotif.end(), ct);
        if (count > 0)
        {
            nCt = std::round(nCt / count);
        }

        py::dict motifOut;
        motifOut["center"] = ct;
        motifOut["motifs"] = sortMotif;
        motifOut["n_center_motif"] = nMotifCt;
        motifOut["n_center"] = nCt;
        motifOut["n_motif"] = nMotifLabels;
        motifsOut.append(motifOut);
    }

    return motifsOut;
}

py::list SpatialDataSingle::motifEnrichmentDist(
    const std::string &ct,
    const py::list &motifs,
    float radius,
    float minSupport,
    bool disDuplicates,
    int minSize)
{
    Idxs idxs(this->coordinates.shape(0));
    std::vector<std::vector<float>> dists(this->coordinates.shape(0));

    auto coords = this->coordinates.template unchecked<2>();
    for (ssize_t i = 0; i < this->coordinates.shape(0); i++)
    {
        std::vector<int> pointIdxs;
        std::vector<float> pointDists;
        pcl::PointXY searchPoint{coords(i, 0), coords(i, 1)};
        kdtree.radiusSearch(searchPoint, radius, pointIdxs, pointDists);
        idxs[i] = std::move(pointIdxs);
        dists[i] = std::move(pointDists);
    }

    std::vector<int> cellTypeIndxs;
    Positions cellTypePos;
    for (size_t i = 0; i < this->labels.size(); i++)
    {
        if (this->labels[i] == ct)
        {
            cellTypeIndxs.push_back(i);
            cellTypePos.push_back({coords(i, 0), coords(i, 1)});
        }
    }

    // Get motifs
    py::list motifsOut;
    py::list motifsToUse;

    if (motifs.size() == 0)
    {
        std::tuple<std::set<Pattern>, std::vector<std::map<Item, int>>, Idxs> FPTreeOut = buildFPTreeDist(cellTypePos, radius, minSupport, disDuplicates, true, minSize);
        std::set<Pattern> frequentPatterns = std::get<0>(FPTreeOut);
        for (const auto &pattern : frequentPatterns)
        {
            py::list itemset;
            for (const auto &item : pattern.first)
            {
                itemset.append(item);
            }
            motifsToUse.append(itemset);
        }
    }
    else
    {
        motifsToUse = motifs;
    }

    if (motifsToUse.size() == 0)
    {
        throw std::runtime_error("No frequent patterns were found. Please lower support threshold.");
    }

    for (const auto &motif : motifsToUse)
    {
        std::vector<Item> motifVec;
        for (const auto &item : motif)
        {
            motifVec.push_back(item.cast<Item>());
        }

        auto sortMotif = motifVec;
        std::sort(sortMotif.begin(), sortMotif.end());

        auto result = motifCalculation(sortMotif, idxs, dists, this->labels, cellTypeIndxs, 100000000);
        int nMotifCt = result.first;
        int nMotifLabels = result.second;

        int nCt = cellTypeIndxs.size();
        int count = std::count(sortMotif.begin(), sortMotif.end(), ct);
        if (count > 0)
        {
            nCt = std::round(nCt / count);
        }

        py::dict motifOut;
        motifOut["center"] = ct;
        motifOut["motifs"] = sortMotif;
        motifOut["n_center_motif"] = nMotifCt;
        motifOut["n_center"] = nCt;
        motifOut["n_motif"] = nMotifLabels;
        motifsOut.append(motifOut);
    }
    return motifsOut;
}

// PYBIND11_MODULE(spatial_module, m)
// {
//     py::class_<SpatialDataSingle>(m, "SpatialDataSingle")
//         .def(py::init<>())
//         .def("set_data", &SpatialDataSingle::setData, "Set the spatial coordinates and labels")
//         .def("get_coordinates", &SpatialDataSingle::getCoordinates, "Get the spatial coordinates")
//         // .def("add_point", &SpatialDataSingle::addPoint, "Add a single point")
//         // .def("add_points", &SpatialDataSingle::addPoints, "Add multiple points")
//         .def("build_kdtree", &SpatialDataSingle::buildKDTree, "Build the KD-tree")
//         .def("radius_search", &SpatialDataSingle::radiusSearch,
//              py::arg("cell_pos"),
//              py::arg("radius") = 100.0,
//              "Get radius search indices by kdtree")
//         .def("build_fptree_knn", py::overload_cast<const py::object &, int, float, bool, bool, float>(&SpatialDataSingle::buildFPTreeKNN),
//              py::arg("cell_pos"),
//              py::arg("k") = 30,
//              py::arg("min_support") = 0.5,
//              py::arg("dis_duplicates") = false,
//              py::arg("if_max") = true,
//              py::arg("max_dist") = 500.0,
//              "Build FP-tree using KNN")
//         .def("build_fptree_dist", py::overload_cast<const py::object &, float, float, bool, bool, int>(&SpatialDataSingle::buildFPTreeDist),
//              py::arg("cell_pos"),
//              py::arg("radius") = 100.0,
//              py::arg("min_support") = 0.5,
//              py::arg("dis_duplicates") = false,
//              py::arg("if_max") = true,
//              py::arg("min_size") = 0,
//              "Build FP-tree using distance")
//         .def("motif_enrichment_knn", &SpatialDataSingle::motifEnrichmentKNN,
//              py::arg("ct"),
//              py::arg("motifs"),
//              py::arg("k") = 30,
//              py::arg("min_support") = 0.5,
//              py::arg("dis_duplicates") = false,
//              py::arg("max_dist") = 200.0,
//              "Perform motif enrichment using KNN")
//         .def("motif_enrichment_dist", &SpatialDataSingle::motifEnrichmentDist,
//              py::arg("ct"),
//              py::arg("motifs"),
//              py::arg("radius") = 100.0,
//              py::arg("min_support") = 0.5,
//              py::arg("dis_duplicates") = false,
//              py::arg("min_size") = 0,
//              "Perform motif enrichment using distance");
// }