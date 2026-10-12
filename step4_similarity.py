
import math


def dot(a, b):
    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def magnitude(a):
    total = 0

    for value in a:
        total += value * value

    return math.sqrt(total)


def cosine_similarity(a, b):
    denominator = magnitude(a) * magnitude(b)

    if denominator == 0:
        return 0.0

    return dot(a, b) / denominator


if __name__ == "__main__":
    a = [2, 3]
    b = [4, 5]

    print("Dot product of a and b:", dot(a, b))
    print("Magnitude:", magnitude([3, 4]))
    print("Cosine similarity of a and b:",
          cosine_similarity([1, 2], [2, 4]))
