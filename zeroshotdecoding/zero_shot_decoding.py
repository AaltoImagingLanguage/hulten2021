# encoding: utf-8
"""
Perform Zero-shot decoding.

In zero-shot learning, the goal is to learn a classifier f : X → Y that must
predict novel values of Y that were omitted from the training set. To achieve
this, we define the notion of a semantic output code classifier (SOC) which
utilizes a knowledge base of semantic properties of Y to extrapolate to novel
classes.

In other words:
Given a semantic encoding of a large set of concept classes, can we build a
classifier to recognize classes that were omitted from the training set?

References
----------
Palatucci, M., Hinton, G., Pomerleau, D., Mitchell, T. M. (2009). Zero-shot
learning with semantic output codes. In Y. Bengio, D. Schuurmans, J. D.
Lafferty, C. K. I. Williams, & A. Culotta (Eds.), Advances in Neural
Information Processing Systems 22 (pp. 1410–1418). Curran Associates, Inc.
"""
from __future__ import division

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from scipy.misc import comb
from sklearn.cross_validation import LeavePOut
import pandas

from .ridge import RidgeGCV


def _leave_two_out_accuracy(ground_truth, predictions, metric='sqeuclidean'):
    """Compute classification accuracy given two examples.

    The classification task is to tell which example is which.

    A distance metric is used to determine which set of predicted targets best
    fits the ground truth of example 1 and which best fits the ground truth of
    example 2. The accuracy can therefore be either 100% or 0%.

    Parameters
    ----------
    ground_truth : 1D numpy array (2,) | 2D numpy array (2, n_targets)
        The real targets of the two examples.
    predictions : 1D numpy array (2,) | 2D numpy array (2, n_targets)
        The predicted targets for the two examples.
    metric : str
        The distance metric to use.

    Returns
    -------
    accuracy : float
        Either 1 (correct) or 0 (incorrect).
    """
    assert len(ground_truth) == 2 and len(predictions) == 2

    # Make sure these are 2D matrices
    if ground_truth.ndim == 1:
        ground_truth = ground_truth[:, np.newaxis]
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    # Compute distances
    D = cdist(predictions, ground_truth, metric=metric)

    # Do the predicted targets match the ground truth targets?
    if (D[0, 0] + D[1, 1]) < (D[0, 1] + D[1, 0]):
        return 1.
    else:
        return 0.


def _patterns_from_weights(X, W, y_hat):
    """Compute activity patterns from regression weights.

    Regression weights originating from multivariage regression cannot be
    directly interpreted. Instead, they must be multiplied with the covariance
    matrix of the data in order to yield 'patterns' instead of 'filters' [1].
    The resulting patterns can be interpreted.

    Parameters
    ----------
    X : 2D array (n_samples, n_features)
        The data the learning was performed on.
    W : 1D array (n_features)
        The regression weights (may include the 1's added for learning the bias).
    y_hat : 2D array (n_samples, n_targets)
        The targets that where predicted from the data (not the ground thruth!).

    Returns
    -------
    P : 2D array (n_samples, n_features)
        The activity patterns.

    References
    ----------
    [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D.,
    Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight
    vectors of linear models in multivariate neuroimaging. NeuroImage, 87,
    96–110.  http://doi.org/10.1016/j.neuroimage.2013.10.067
    """
    # Check input dimensions
    n_samples, n_features = X.shape
    if W.shape[1] != n_features:
        raise ValueError("The number of weights doesn't match the number of features.")
    assert y_hat.shape[0] == n_samples

    # Compute inverse covariance of the predicted targets
    SS = (y_hat - y_hat.mean(axis=0)) / np.sqrt(n_samples - 1)
    S_cov_i = np.linalg.pinv(SS.T.dot(SS))

    # Compute patterns
    XX = (X - X.mean(axis=0)) / np.sqrt(n_samples - 1)

    # Rewrite equation (6) from Haufe2014 to be more computationally friendly:
    # P = cov_X * W * cov_S^(-1)
    # P = (X.T * X) * W * cov_S^(-1)
    # P = X.T * (X * W) * cov_S^(-1)

    P = XX.T.dot(XX.dot(W.T))
    P = P.dot(S_cov_i).T

    return P


def zero_shot_decoding(X, y, verbose=True, break_after=None, normalize=True,
                       model=None, metric='sqeuclidean'):
    """Perform zero shot decoding.

    Zero-shot learning refers to a specific way to evaluate a model. Given two
    examples (rows of X) and two labels (rows of y), the task is to match each
    example to the right label. Note that this is a binary choice: when
    example 1 belongs to label 1, example 2 necessarily belongs to label 2;
    or vice versa.

    This function trains and evaluates a given model (which defaults to
    linear ridge regression) on each possible pair of samples, using a
    leave-two-out cross validation scheme.

    Parameters
    ----------
    X : 2D array (n_samples, n_features)
        The data to perform learning on.
    y : 2D array (n_samples, n_targets)
        The targets to learn from the data.
    verbose : bool
        Whether to display a progress bar or not. Defaults to False.
    break_after : int | None
        Break after this many pairwise comparisons. Useful for debugging.
        Defaults to None, which means all comparisons will be performed.
    normalize : bool
        Whether to normalize X and y (normalize along each column).
        Defaults to True (you normally always want to do this).
    model : scikit-learn regression model | None
        The model to use. If None, the RidgeGCV model, given in ridge.py
        will be used. You can also pass a scikit-learn pipeline
        here, which is useful if you want to add some preprocessing steps that
        should be done at each iteration of the leave-two-out loop.
    metric : str
        The distance metric to use. Any distance implemented in SciPy's
        spatial.distance module is supported. See the docstring of
        scipy.spatial.distance.pdict for the exhaustive list of possitble
        metrics. Here are some of the more useful ones:
        'euclidean' - Euclidean distance
        'sqeuclidean' - Squared euclidean distance (the default)
        'correlation' - Pearson correlation
        'cosine' - Cosine similarity

    Returns
    -------
    pairwise_accuracies : 2D array of bool (n_examples, n_examples)
        Matrix containing the results for all pairwise comparisons.
        To tell whether, given 2 examples, the model could correctly tell which
        was which, you can perform a lookup in this matrix:
        pairwise_accuracies[word1_index, word2_index]
        (where word1_index < word2_index)
    model : a scikit-learn regression model
        The model trained on all samples. This is the same model as specied
        in the `model` input parameter. You can use this object to obtain the
        regression weights and such.
    target_scores : 1D array (n_targets,)
        For each target, a score between 0 and 1 of how well it could be
        predicted from the data.
    predicted_y : 2D array (n_items, n_ginter_features)
        The predicted targets using a model trained on all samples.
    patterns : 2D array (n_samples, n_features)
        Activity patterns that can be interpreted.

    Notes
    -----
     - Currently, each sample must belong to a unique label.
     - The zero-shot-learning approach works best when labels have multiple
       targets (i.e. y has multiple columns).

    See also
    --------
    The predictions_breakdown function can be used to compute accuracies given
    the pairwise_accuracies matrix produced by this function.
    """
    assert len(X) == len(y)
    n_samples, n_features = X.shape
    _, n_targets = y.shape

    # Normalize the features and the targets
    if normalize:
        X = zscore(X, axis=0, ddof=1)
        y = zscore(y, axis=0, ddof=1)

    # Results will be saved to this pairwise matrix
    pairwise_accuracies = np.zeros((n_samples, n_samples), dtype=np.bool)

    # model to use
    if model is None:
        #alphas = [.0000001, .000001, .00001, .0001, .001, .01, .1, .5, 1, 5,
        #          10, 50, 100, 500, 1000, 10000, 20000, 50000, 100000, 500000,
        #          1000000, 5000000, 10000000]
        model = RidgeGCV(alphas=np.logspace(-5, 5, 100), alpha_per_target=True,
                         fit_intercept=True, normalize=True,
                         store_cv_values=True)
        # model = Ridge(alpha=1E10)

    # Do leave-2-out crossvalidation
    splits = list(LeavePOut(n_samples, 2))
    n_splits = len(splits)

    if verbose:
        from tqdm import tqdm
        pb = tqdm(total=n_splits, unit='iterations')

    # Store predictions
    predicted_y = np.zeros((n_samples, n_targets))

    i = 0
    for train_index, test_index in splits:
        model.fit(X[train_index], y[train_index])
        prediction = model.predict(X[test_index])
        accuracy = _leave_two_out_accuracy(y[test_index], prediction, metric=metric)

        # Store the result in the upper triangle of the pairwise matrix
        if test_index[0] < test_index[1]:
            pairwise_accuracies[test_index[0], test_index[1]] = accuracy
        else:
            pairwise_accuracies[test_index[1], test_index[0]] = accuracy

        # Store prediction
        predicted_y[test_index[0]] += prediction[0]
        predicted_y[test_index[1]] += prediction[1]

        if verbose:
            pb.update(1)

        i += 1
        if break_after is not None and i >= break_after:
            break

    if verbose:
        pb.close()

    # Fit model to all data. This is the model that gets returned
    model.fit(X, y)

    # Compute the mean of the predicted targets (they have been computed twice
    # due to the leave-two-out scheme).
    predicted_y /= 2

    # Calculate the "coefficient of determination" (Eq. 4 in Sudre et al. 2012)
    # This value represents how well we are able to predict a target (from 0
    # to 1; 1 implying perfect prediction of the targets).
    # target_scores = 1 - (np.mean(model.cv_values_.min(axis=2), axis=0) /
    #                      np.mean(y ** 2, axis=0))
    target_scores = 1 - (np.mean((y - predicted_y) ** 2, axis=0) /
                         np.mean(y ** 2, axis=0))

    # Multivariate regression weights cannot be interpreted. Compute patterns
    # that *can* be. (Drop bias terms from X and model.coef_)
    patterns = _patterns_from_weights(X, model.coef_, predicted_y)

    return pairwise_accuracies, model, target_scores, predicted_y, patterns


def predictions_breakdown(accuracies, category_assignments, category_labels,
                          verbose=True):
    """
    Compute and display various statistics regarding the accuracy of the
    predictions made by the zero-shot learning approach.

    Parameters
    ---------
    accuracies: 2D array (n_examples, n_examples)
        The pairwise accuracy matrix.
    category_assignments : 1D array (n_examples,)
        For each example, the number of the category it belongs to.
    category_labels : dict (int -> str)
        A dictionary mapping each category number to a string label.
    verbose : bool
        Whether to print out the results as they are computed.
        Defaults to True.

    Returns
    -------
    results : instance of DataFrame
        A Pandas DataFrame object that contains the results.

    See also
    --------
    zero_shot_decoding
    """
    # Sanity checks
    assert accuracies.shape[0] == accuracies.shape[1]

    n_examples = accuracies.shape[0]
    assert len(category_assignments) == n_examples, (
        'number of rows in accuracies should match the length of '
        'category_assignments')

    category_codes = np.unique(category_assignments)
    n_categories = len(category_codes)

    # Results are collected in this list as (key, value) tuples.
    # (keys are (str, str) tuples themselves)
    results = []

    if verbose:
        print('\nAccuracy\t#Comp.\tDescription')
        print('-----------------------------------')

    mask = np.triu(np.ones_like(accuracies, dtype=np.bool), 1)
    accuracy = np.mean(accuracies[mask])
    n_comparisons = len(accuracies[mask])
    results.append((('all', 'all'), (accuracy, n_comparisons)))
    if verbose:
        print('%f\t%d\t%s' % (accuracy, n_comparisons, 'Overall accuracy'))

    # Within category accuracies
    within_cat_accuracies = np.zeros(n_categories)
    n_within_comparisons = np.zeros(n_categories)
    for i, category_code in enumerate(category_codes):
        words_in_cat = category_assignments == category_code
        trimmed_accuracies = accuracies[words_in_cat, :][:, words_in_cat]
        mask = np.triu(np.ones_like(trimmed_accuracies, dtype=np.bool), 1)
        within_cat_accuracies[i] = np.mean(trimmed_accuracies[mask])
        n_within_comparisons[i] = len(trimmed_accuracies[mask])
        accuracy = within_cat_accuracies[i]
        results.append((
            (category_labels[category_code], category_labels[category_code]),
            (accuracy, n_within_comparisons[i])
        ))
        if verbose:
            print('%f\t%d\tWithin %s' % (accuracy,
                                         n_within_comparisons[i],
                                         category_labels[category_code]))

    # Between category accuracies
    between_cat_accuracies = np.zeros(int(comb(n_categories, 2)))
    n_between_comparisons = np.zeros(int(comb(n_categories, 2)))
    ind = 0
    for i in range(n_categories):
        for j in range(i + 1, n_categories):
            words_in_cat1 = category_assignments == category_codes[i]
            words_in_cat2 = category_assignments == category_codes[j]
            mask = np.zeros_like(accuracies, dtype=np.bool)
            mask[np.dot(
                words_in_cat1[:, np.newaxis],
                words_in_cat2[np.newaxis, :]
            )] = True
            mask = np.triu(mask, 1)
            between_cat_accuracies[ind] = np.mean(accuracies[mask])
            n_between_comparisons[ind] = len(accuracies[mask])
            accuracy = between_cat_accuracies[ind]
            results.append((
                (category_labels[category_codes[i]],
                 category_labels[category_codes[j]]),
                (accuracy, n_between_comparisons[ind])
            ))
            if verbose:
                print('%f\t%d\t%s vs %s' % (accuracy,
                                            n_between_comparisons[ind],
                                            category_labels[category_codes[i]],
                                            category_labels[category_codes[j]]))
            ind = ind + 1

    overall_within_cat_accuracy = (
        np.sum(within_cat_accuracies * n_within_comparisons) /
        np.sum(n_within_comparisons)
    )
    overall_between_cat_accuracy = (
        np.sum(between_cat_accuracies * n_between_comparisons) /
        np.sum(n_between_comparisons)
    )
    results.append((('category', 'within'),
                    (overall_within_cat_accuracy, np.sum(n_within_comparisons))))
    results.append((('category', 'between'),
                    (overall_between_cat_accuracy, np.sum(n_between_comparisons))))

    if verbose:
        print('%f\t%d\tOverall within category' % (
            overall_within_cat_accuracy, np.sum(n_within_comparisons)))
        print('%f\t%d\tOverall between category' % (
            overall_between_cat_accuracy, np.sum(n_between_comparisons)))

    return pandas.DataFrame(
        index=pandas.MultiIndex.from_tuples([x[0] for x in results],
                                            names=['X', 'Y']),
        data=[x[1] for x in results],
        columns=['accuracy', 'n_comparisons'],
    )
