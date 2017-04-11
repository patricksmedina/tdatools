import numpy as np

def persistent_entropy(L):
    """Compute entropy for a set of persistence lifetimes.

    Arguments
        1. L: numpy array of lifetimes

    Returns
        1. entropy (float)

    Raises
        None
    """
    p = np.divide(L, L.sum())
    Ef = np.multiply(p, np.log(p))
    return(-Ef.sum())


def filter_diagram_w_entropy(L):
    """Use persistence entropy to remove noise from a persistence
    diagram.

    """

    T = L[0]
    r = L[-1]
    alpha = r / T
    L0 = L[1:-1]

    while True:
        n0 = len(L0) + 2
        Q = alpha * n0 * (alpha - 1 - np.log(alpha)) / (alpha - 1) ** 2

        for i in np.arange(1, n0):
            # compute entropy ratio
            Lp = np.append(L0, [r, T])
            C = _compute_fixed_entropy_ratio(Lp, i)

            if C < 1:
                break
        print(i)
        L0 = L0[:i]
        if Q >= i:
            break
        a = raw_input("Keep going? ")

    L0 = np.append(L0, [T])
    return(L0.sort()[::-1])


def _compute_fixed_entropy_ratio(L, i):
    """

    """
    # compute at step i - 1
    pi = L[i-1:].sum()
    ei = persistent_entropy(L[i-1:])

    # compute at step i
    Pi = L[i:].sum()
    Ei = persistent_entropy(L[i+1:])

    # compute C
    C = pi + (i - 1) * pi / np.exp(ei)
    C /= Pi + i * Pi / np.exp(Ei)

    return(C)


if __name__ == "__main__":
    # generate fake persistence data
    np.random.seed(1034934)
    L = np.concatenate((np.random.uniform(low = 0, high = 0.5, size = 5),
                       np.random.uniform(low = 1, high = 1.5, size = 3)))

    L.sort()
    L = L[::-1]

    # print results
    print(L)
    print(filter_diagram_w_entropy(L))
