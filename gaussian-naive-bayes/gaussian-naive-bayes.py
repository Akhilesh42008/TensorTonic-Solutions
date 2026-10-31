import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    classes = sorted(set(y_train))
    n = len(y_train)
    n_features = len(X_train[0])
    stats = {}
    for c in classes:
        samples = [X_train[i] for i in range(n) if y_train[i] == c]
        prior = len(samples) / n
        means = []
        variances = []
        for j in range(n_features):
            values = [sample[j] for sample in samples]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            means.append(mean)
            variances.append(max(variance, 1e-9))
        stats[c] = (prior, means, variances)
    predictions = []
    for sample in X_test:
        best_class = None
        best_score = float('-inf')
        for c in classes:
            prior, means, variances = stats[c]
            score = math.log(prior)
            for j in range(n_features):
                x = sample[j]
                mean = means[j]
                variance = variances[j]
                score += -0.5 * math.log(2 * math.pi * variance)
                score += -((x - mean) ** 2) / (2 * variance)
            if score > best_score:
                best_score = score
                best_class = c
        predictions.append(best_class)
    return predictions

    pass