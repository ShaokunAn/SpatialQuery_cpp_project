#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>
#include <string>

#include "typedef.h"
#include "utils.h"

Transaction distinguishDuplicates(const Transaction &transaction)
{
    std::unordered_map<std::string, int> occurrenceCount;
    Transaction transWithSuffixes;
    transWithSuffixes.reserve(transaction.size());

    for (const auto &item : transaction)
    {
        // Increment the occurrence count for this item
        int &count = occurrenceCount[item];
        count++;

        // Append this item with its current count as a suffix
        transWithSuffixes.push_back(item + "_" + std::to_string(count));
    }

    return transWithSuffixes;
}

// std::set<Pattern> findMaximalPatterns(const std::set<Pattern> &patterns)
// {
//     std::set<Itemset> itemsets;
//     for (const auto &pattern : patterns)
//     {
//         itemsets.insert(pattern.first);
//     }

//     std::set<Itemset> subsets;
//     for (const auto &itemset : itemsets)
//     {
//         for (size_t r = 1; r < itemset.size(); ++r)
//         {
//             std::vector<Itemset> combinations;
//             std::vector<bool> v(itemset.size());
//             std::fill(v.begin(), v.begin() + r, true);

//             do
//             {
//                 Itemset subset;
//                 for (size_t i = 0; i < itemset.size(); ++i)
//                 {
//                     if (v[i])
//                     {
//                         subset.insert(*std::next(itemset.begin(), i));
//                     }
//                 }
//                 combinations.push_back(subset);
//             } while (std::prev_permutation(v.begin(), v.end()));

//             for (const auto &combination : combinations)
//             {
//                 subsets.insert(combination);
//             }
//         }
//     }

//     std::set<Itemset> maximalPatterns;
//     for (const auto &itemset : itemsets)
//     {
//         if (subsets.find(itemset) == subsets.end())
//         {
//             maximalPatterns.insert(itemset);
//         }
//     }

//     std::set<Pattern> maximalPatternsWithSupport;
//     for (const auto &pattern : patterns)
//     {
//         if(maximalPatterns.find(pattern.first)!=maximalPatterns.end())
//         {
//             maximalPatternsWithSupport.insert(pattern);
//         }
//     }

//     return maximalPatternsWithSupport;
// }

std::set<Pattern> findMaximalPatterns(const std::set<Pattern> &patterns)
{
    std::vector<Itemset> maximalPatterns;

    for (const auto &pattern : patterns)
    {
        bool isMaximal = true;
        std::vector<size_t> subsetsToRemove;

        for (size_t i = 0; i < maximalPatterns.size(); ++i)
        {
            const auto maxPattern = maximalPatterns[i];
            if (std::includes(pattern.first.begin(), pattern.first.end(), maxPattern.begin(), maxPattern.end()))
            {
                subsetsToRemove.push_back(i);
            }
            else if (std::includes(maxPattern.begin(), maxPattern.end(), pattern.first.begin(), pattern.first.end()))
            {
                isMaximal = false;
                break;
            }
        }

        for (size_t i = subsetsToRemove.size(); i > 0; --i)
        {
            maximalPatterns.erase(maximalPatterns.begin() + subsetsToRemove[i - 1]);
        }

        if (isMaximal)
        {
            maximalPatterns.push_back(Itemset(pattern.first.begin(), pattern.first.end()));
        }
    }

    std::set<Pattern> maximalPatternsWithSupport;

    for (const auto &pattern : patterns)
    {
        Itemset itemset(pattern.first.begin(), pattern.first.end());
        if (std::find(maximalPatterns.begin(), maximalPatterns.end(), itemset)!=maximalPatterns.end())
        {
            maximalPatternsWithSupport.insert(pattern);
        }
    }

    return maximalPatternsWithSupport;
}

std::set<DuplicatePattern> removeSuffix(const std::set<Pattern> &patterns)
{
    std::set<DuplicatePattern> modifiedPatterns;

    for (const auto &pattern : patterns)
    {
        std::vector<Item> modifiedItemset;
        for (const auto &item : pattern.first)
        {
            size_t pos = item.find('_');
            if (pos != std::string::npos)
            {
                modifiedItemset.push_back(item.substr(0, pos));
            }
            else
            {
                modifiedItemset.push_back(item);
            }
        }

        // Insert the modified itemset along with its original frequency into the new set
        modifiedPatterns.insert({modifiedItemset, pattern.second});
    }

    return modifiedPatterns;
}

bool hasMotif(const std::vector<Item> &motif, const std::vector<Item> &neighborLabels)
{
    std::unordered_map<std::string, int> freqMotif;
    for (const std::string &element : motif)
    {
        freqMotif[element]++;
    }
    for (const std::string &element : neighborLabels)
    {
        auto it = freqMotif.find(element);
        if (it != freqMotif.end())
        {
            it->second--;
            if (it->second == 0)
            {
                freqMotif.erase(it);
                if (freqMotif.empty())
                {
                    return true;
                }
            }
        }
    }
    return false;
}

std::pair<int, int> motifCalculation(
    const std::vector<Item> &motif,
    const Idxs &idxs,
    const std::vector<std::vector<float>> &dists,
    std::vector<std::string> &labels,
    const std::vector<int> &cinds,
    float maxDist)
{
    std::pair<int, int> out;
    float maxDistSquare = maxDist * maxDist;

    int nMotifCt = 0; // nMotifCt is the number of centers nearby specified cell types (motif)
    for (int i : cinds)
    {
        std::vector<int> inds;
        for (size_t ind = 0; ind < dists[i].size(); ++ind)
        {
            if (dists[i][ind] < maxDistSquare)
            {
                inds.push_back(ind);
            }
        }
        if (inds.size() > 0)
        {
            std::vector<std::string> neighborLabels;
            for (size_t j = 1; j < inds.size(); ++j)
            {
                neighborLabels.push_back(labels[idxs[i][inds[j]]]);
            }
            if (hasMotif(motif, neighborLabels))
            {
                nMotifCt++;
            }
        }
    }

    int nMotifLabels = 0; // nMotifLabels is the number of all cell_pos nearby specified motifs
    for (size_t i = 0; i < labels.size(); ++i)
    {
        std::vector<Item> neighborLabels;
        for (size_t j = 1; j < idxs[i].size(); ++j)
        {
            neighborLabels.push_back(labels[idxs[i][j]]);
        }
        if (hasMotif(motif, neighborLabels))
        {
            nMotifLabels++;
        }
    }

    out = std::make_pair(nMotifCt, nMotifLabels);

    return out;
}