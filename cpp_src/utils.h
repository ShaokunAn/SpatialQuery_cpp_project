#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>

#include "typedef.h"

Transaction distinguishDuplicates(const Transaction &transaction);

// Function to find maximal itemsets using the TrieNode
std::set<Pattern> findMaximalPatterns(const std::set<Pattern> &patterns);

std::set<DuplicatePattern> removeSuffix(const std::set<Pattern> &patterns);

bool hasMotif(const std::vector<Item> &motif, const std::vector<Item> &neighborLabels);

std::pair<int, int> motifCalculation(
    const std::vector<Item> &motif,
    const Idxs &idxs,
    const std::vector<std::vector<float>> &dists,
    std::vector<std::string> &labels,
    const std::vector<int> &cinds,
    float maxDist);