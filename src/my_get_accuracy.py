def run(X, y, normalize):
    match = 0
    total = len(X)

    for i in range(0, total):
        if X[i] == y[i]:
            match += 1

    if normalize:
        return 1.0 * match / total

    return match

