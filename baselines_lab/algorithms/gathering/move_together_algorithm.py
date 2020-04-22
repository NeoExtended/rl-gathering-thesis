from algorithms.gathering.iterative_merging import \
    IterativeMostDistancedPairMergingAlgorithm
from algorithms.gathering.pair_merging.merge_via_dynamic_shortest_path import \
    MergeViaDynamicShortestPathAlgorithm
from algorithms.gathering.pair_merging.merge_via_static_shortest_path import \
    MergeViaStaticShortestPathAlgorithm

def DynamicShortestPathFollowingAlgorithm():
    alg = IterativeMostDistancedPairMergingAlgorithm()
    alg.set_merging_algorithm(MergeViaDynamicShortestPathAlgorithm())
    return alg

def StaticShortestPathFollowingAlgorithm():
    alg = IterativeMostDistancedPairMergingAlgorithm()
    alg.set_merging_algorithm(MergeViaStaticShortestPathAlgorithm())
    return alg

