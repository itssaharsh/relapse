"""A two-value view of analyze_map.paired_p for scripts that only need the change and its p-value."""
from analyze_map import paired_p


def paired(a, b, rng):
    delta, _, _, p = paired_p(a, b, rng)
    return delta, p
