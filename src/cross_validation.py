"""
cross_validation.py

Machine Learning - IMAT
ICAI, Universidad Pontificia Comillas

Lydia Ruiz

Description:
Program that performs n-fold cross-validation or Leave One Out (LOO) cross-validation on
a given machine learning model to evaluate its accuracy. The function manually splits the
dataset, trains the model on each fold, and calculates both the mean and standard deviation
of the accuracy scores across all folds.
"""

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.
    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.
    """
    if nFolds == -1:
        nFolds = X.shape[0]
    fold_size = int(X.shape[0] / nFolds)
    accuracy_scores = []
    
    for i in range(nFolds):
        valid_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, X.shape[0]))
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        model.fit(X_train, y_train)
        accuracy = model.score(X_valid, y_valid)
        accuracy_scores.append(accuracy)

    mean_score = sum(accuracy_scores) / len(accuracy_scores)
    std_score = (sum((x - mean_score) ** 2 for x in accuracy_scores) / len(accuracy_scores)) ** 0.5
    
    return mean_score, std_score