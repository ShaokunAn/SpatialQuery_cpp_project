from typing import List


def find_maximal_patterns(patterns: List[List[str]]):
    patterns.sort(key=len, reverse=True)
    maximal_pattern = []

    for pattern in patterns:
        if not any(set(pattern).issubset(other) for other in patterns):
            maximal_pattern.append(pattern)

    return maximal_pattern
