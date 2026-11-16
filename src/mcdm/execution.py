from src.mcdm.enums import(
    WeightingMethod,
    RankingMethod,
    MeanMethod
)

def select_method(registry, method):
    try:
        return registry[method]
    except KeyError as error:
        raise ValueError(
            f"Method {method!r} is not implemented."
        ) from error

def calculate_ahp():
    x = 1

# def
# WEIGHTING_METHODS = {
#     WeightingMethod.AHP: calculate_ahp,
# }

# RANKING_METHODS = {
#     RankingMethod.TOPSIS: calculate_topsis,
# }

# WITHIN_GROUP_METHODS = {
#     MeanMethod.GEOMETRIC_MEAN: aggregate_geometric,
# }

# BETWEEN_GROUP_METHODS = {
#     MeanMethod.WEIGHTED_GEOMETRIC_MEAN: aggregate_weighted_geometric,
# }