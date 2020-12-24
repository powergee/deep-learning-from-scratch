def nonBiasedPerceptron(w1, w2, theta, x1, x2):
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def biasedPerceptron(w1, w2, b, x1, x2):
    tmp = x1*w1 + x2*w2 + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# AND
def nonBiasedAnd(x1, x2):
    return nonBiasedPerceptron(0.5, 0.5, 0.7, x1, x2)


def biasedAnd(x1, x2):
    return biasedPerceptron(0.5, 0.5, -0.7, x1, x2)

# OR
def nonBiasedOr(x1, x2):
    return nonBiasedPerceptron(0.5, 0.5, 0.2, x1, x2)


def biasedOr(x1, x2):
    return biasedPerceptron(0.5, 0.5, -0.2, x1, x2)

# NAND
def nonBiasedNand(x1, x2):
    return nonBiasedPerceptron(-0.5, -0.5, -0.7, x1, x2)


def biasedNand(x1, x2):
    return biasedPerceptron(-0.5, -0.5, 0.7, x1, x2)


# XOR
def nonBiasedXor(x1, x2):
    return nonBiasedAnd(nonBiasedNand(x1, x2), nonBiasedOr(x1, x2))


def biasedXor(x1, x2):
    return biasedAnd(biasedNand(x1, x2), biasedOr(x1, x2))


def printTable(name, perc):
    print(name.center(32, ' '))
    print('-' * 32)
    for i in range(0, 2):
        for j in range(0, 2):
            print(str(i).center(10, ' '), end="")
            print("|", end="")
            print(str(j).center(10, ' '), end="")
            print("|", end="")
            print(str(perc(i, j)).center(10, ' '))
    print()

if __name__ == "__main__":
    printTable("Non-Biased AND", nonBiasedAnd)
    printTable("Biased AND", biasedAnd)
    printTable("Non-Biased OR", nonBiasedOr)
    printTable("Biased OR", biasedOr)
    printTable("Non-Biased NAND", nonBiasedNand)
    printTable("Biased NAND", biasedNand)
    printTable("Non-Biased XOR", nonBiasedXor)
    printTable("Biased XOR", biasedXor)