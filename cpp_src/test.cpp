#include <iostream>
#include "FPGrowth.h"

// Example of finding frequent patterns using Nicos' codes.
int main() {
    // Transactions dataset
    std::vector<Transaction> transactions = {
        {"apple", "banana", "milk"},
        {"banana", "beer", "chips"},
        {"apple", "banana", "beer"},
        {"apple", "chips"},
        {"banana", "beer"}
    };

    // Set the minimum support threshold
    uint64_t minimum_support_threshold = 2;

    // Create the FPTree
    FPTree fptree(transactions, minimum_support_threshold);

    // Find frequent patterns
    auto patterns = fptree_growth(fptree);

    // Display the frequent patterns
    std::cout << "Frequent patterns (min support " << minimum_support_threshold << "):" << std::endl;
    for (const auto& pattern : patterns) {
        std::cout << "{ ";
        for (const auto& item : pattern.first) {
            std::cout << item << " ";
        }
        std::cout << "}: " << pattern.second << std::endl;
    }

    return 0;
}
