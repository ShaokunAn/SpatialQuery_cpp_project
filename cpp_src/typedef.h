#include <unordered_set>
using Item = std::string;
using Transaction = std::vector<Item>;
using Itemset = std::vector<Item>;
using DuplicateItemset = std::vector<Item>;
using Pattern = std::pair<std::set<Item>, uint64_t>;
using DuplicatePattern = std::pair<DuplicateItemset, uint64_t>;
using Idxs = std::vector<std::vector<int>>;
using Positions = std::vector<std::vector<float>>;
