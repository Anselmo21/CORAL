# # for the paper
# # calculate the softmax probability and entropy of a given ranking
# # uses the softmax function and the entropy formula

from search.heuristics import softmax_entropy, shannon_entropy
import numpy as np

ranking = ["a", "b", "c"]
# scores = {
#     "a": 6,
#     "b": 3,
#     "c": 3,
# }
scores_list = [
    [6, 3, 3],
    [6, 5, 1],
    [5, 5, 2],
    [4, 4, 4],
]

for scores in scores_list:
    print("scores:", scores)
    scores = dict(zip(ranking, scores))
    print("\tshannon entropy:", shannon_entropy(scores))
    print("\tsoftmax entropy:", softmax_entropy(scores))
