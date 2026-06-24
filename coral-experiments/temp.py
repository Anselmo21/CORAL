# # for the paper
# # calculate the softmax probability and entropy of a given ranking
# # uses the softmax function and the entropy formula

from search.heuristics import softmax_entropy
import numpy as np

ranking = ["a", "b", "c"]
# scores = {
#     "a": 6,
#     "b": 3,
#     "c": 3,
# }
scores_list = [
    [18, 6, 12],
    [3, 1, 2],
]
# # scores = [18, 6, 12]
# scores = [3, 1, 2]

for scores in scores_list:
    scores = dict(zip(ranking, scores))
    print(softmax_entropy(scores))

    # by hand
    # 1. calculate the softmax probability of each item
    scores = list(scores.values())
    def softmax(scores):
        return np.exp(scores) / np.sum(np.exp(scores))

    # def prob(scores):
    #     return scores / np.sum(scores)

    print("softmax probabilities:", softmax(scores))
    # print("probabilities:", prob(scores))

    # 2. calculate the entropy of the probabilities
    def entropy(probabilities):
        return -np.sum(probabilities * np.log2(probabilities))

    print("entropy:", entropy(softmax(scores)))

