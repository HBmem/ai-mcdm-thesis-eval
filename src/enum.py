"""

"""
from __future__ import annotations

from enum import Enum

class WeightingMethod(str, Enum):
    """
    Available methods for weighting criteria.
    """

    AHP = "ahp"
    FUZZY_AHP = "fuzzy_ahp"

class RankingMethod(str, Enum):
    """
    Available methods for ranking alternatives.
    """

    TOPSIS = "topsis"
    FUZZY_TOPSIS = "fuzzy_topsis"

class CriterionDirection(str, Enum):
    """
    Indicates whether higher or lower values are preferred for a criterion.
    """

    MAXIMIZE = "max"
    MINIMIZE = "min"

class PairwisePreference(str, Enum):
    """
    Semantic representation of an AHP pairwise comparison response.

    Uses Saaty's conventional 1–9 AHP intensity scale
    """
    EXTREME_IMPORTANCE_RECIPROCAL = -4
    VERY_STRONG_IMPORTANCE_RECIPROCAL = -3
    STRONG_IMPORTANCE_RECIPROCAL = -2
    MODERATE_IMPORTANCE_RECIPROCAL = -1
    EQUAL_IMPORTANCE = 0
    MODERATE_IMPORTANCE = 1
    STRONG_IMPORTANCE = 2
    VERY_STRONG_IMPORTANCE = 3
    EXTREME_IMPORTANCE = 4

    @property
    def ahp_scale(self) -> float:
        """
        Return the numeric AHP ratio represented by this preference.
        """
        mapping = {
            PairwisePreference.EXTREME_IMPORTANCE_RECIPROCAL: 1.0 / 9.0,
            PairwisePreference.VERY_STRONG_IMPORTANCE_RECIPROCAL: 1.0 / 7.0,
            PairwisePreference.STRONG_IMPORTANCE_RECIPROCAL: 1.0 / 5.0,
            PairwisePreference.MODERATE_IMPORTANCE_RECIPROCAL: 1.0 / 3.0,
            PairwisePreference.EQUAL_IMPORTANCE: 1.0,
            PairwisePreference.MODERATE_IMPORTANCE: 3.0,
            PairwisePreference.STRONG_IMPORTANCE: 5.0,
            PairwisePreference.VERY_STRONG_IMPORTANCE: 7.0,
            PairwisePreference.EXTREME_IMPORTANCE: 9.0,
        }
        return mapping[self]