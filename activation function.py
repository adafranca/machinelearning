import numpy as np


def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)


def main():
    scores = [3.0, 1.0, 0.2]
    print(softmax(scores))


main()