from typing import List


def find_maximal_patterns(patterns: List[List[str]]):
    patterns.sort(key=len, reverse=True)
    maximal_pattern = []

    for pattern in patterns:
        is_subset = False
        for other_pattern in patterns:
            if pattern != other_pattern and set(pattern).issubset(set(other_pattern)):
                is_subset = True
                break
        if not is_subset:
            maximal_pattern.append(pattern)

    return maximal_pattern


