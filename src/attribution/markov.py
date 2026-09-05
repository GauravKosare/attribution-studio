"""Markov chain attribution with removal effect — docs/04_attribution_methodology.md section B.

Builds the transition graph dynamically from whatever channels appear in the input
journeys. No fixed channel list, no fixed graph size.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

START, CONVERSION, NULL = "Start", "Conversion", "Null"


def _build_graph(journeys: pd.DataFrame) -> nx.DiGraph:
    counts = defaultdict(lambda: defaultdict(int))
    for _, row in journeys.iterrows():
        path = row["path"]
        if not path:
            continue
        seq = [START] + path + [CONVERSION if row["converted"] else NULL]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1

    graph = nx.DiGraph()
    for src, targets in counts.items():
        total = sum(targets.values())
        for dst, c in targets.items():
            graph.add_edge(src, dst, weight=c / total)
    # absorbing states loop to themselves
    for absorbing in (CONVERSION, NULL):
        if absorbing in graph:
            graph.add_edge(absorbing, absorbing, weight=1.0)
    return graph


def _total_conversion_probability(graph: nx.DiGraph) -> float:
    """Solve the absorbing Markov chain exactly via the fundamental matrix."""
    states = list(graph.nodes)
    if START not in states or CONVERSION not in states:
        return 0.0

    transient = [s for s in states if s not in (CONVERSION, NULL)]
    absorbing = [s for s in (CONVERSION, NULL) if s in states]
    idx = {s: i for i, s in enumerate(transient)}

    n = len(transient)
    Q = np.zeros((n, n))
    R = np.zeros((n, len(absorbing)))
    for s in transient:
        for _, dst, data in graph.out_edges(s, data=True):
            w = data["weight"]
            if dst in idx:
                Q[idx[s], idx[dst]] += w
            elif dst in absorbing:
                R[idx[s], absorbing.index(dst)] += w

    try:
        N = np.linalg.inv(np.eye(n) - Q)
    except np.linalg.LinAlgError:
        return 0.0
    B = N @ R  # absorption probabilities from each transient state

    start_i = idx[START]
    conv_col = absorbing.index(CONVERSION) if CONVERSION in absorbing else None
    return float(B[start_i, conv_col]) if conv_col is not None else 0.0


def removal_effect(journeys: pd.DataFrame) -> dict[str, float]:
    """Removal-effect credit share per channel, normalized to sum to 1."""
    graph = _build_graph(journeys)
    base_p = _total_conversion_probability(graph)
    channels = sorted({ch for path in journeys["path"] for ch in path})

    if base_p == 0 or not channels:
        return {ch: 0.0 for ch in channels}

    effects = {}
    for ch in channels:
        g2 = graph.copy()
        if ch in g2:
            in_edges = list(g2.in_edges(ch, data=True))
            g2.remove_node(ch)
            # any transition that would have gone through ch is redirected to Null
            for src, _, data in in_edges:
                if src in g2:
                    g2.add_edge(src, NULL, weight=g2.get_edge_data(src, NULL, {"weight": 0})["weight"] + data["weight"])
        p_without = _total_conversion_probability(g2)
        effects[ch] = max(0.0, base_p - p_without)

    total = sum(effects.values())
    if total == 0:
        return {ch: 0.0 for ch in channels}
    return {ch: v / total for ch, v in effects.items()}


def run(journeys: pd.DataFrame) -> pd.DataFrame:
    credit = removal_effect(journeys)
    return pd.DataFrame(
        [{"channel": ch, "model": "markov", "credit_share": v} for ch, v in credit.items()]
    )
