"""Map reference words to recognized word-timed segments without inventing times."""
from __future__ import annotations

import re
from typing import Any


def normalize_word(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9']+", str(value).lower()))


def align_words(expected: list[str], recognized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Global edit-distance alignment; substitutions retain the ASR time span."""
    exp = [normalize_word(word) for word in expected]
    hyp = [normalize_word(item.get("word", "")) for item in recognized]
    n, m = len(exp), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "delete"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (exp[i - 1] != hyp[j - 1])
            options = [(sub, "pair"), (cost[i - 1][j] + 1, "delete"),
                       (cost[i][j - 1] + 1, "insert")]
            cost[i][j], back[i][j] = min(options, key=lambda item: item[0])
    result = [None] * n
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step == "pair":
            result[i - 1] = {"recognized_index": j - 1,
                             "text_match": exp[i - 1] == hyp[j - 1]}
            i -= 1
            j -= 1
        elif step == "delete":
            result[i - 1] = {"recognized_index": None, "text_match": False}
            i -= 1
        else:
            j -= 1
    return result

