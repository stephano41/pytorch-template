import itertools
from functools import partial
from typing import List

import numpy as np
import scipy
from mlxtend.evaluate import cochrans_q
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split


def _cv_5x2(model, X, Y, randints):
    assert len(randints) == 5

    outputs = []
    targets = []
    for randint in randints:
        X_1, X_2, y_1, y_2 = train_test_split(X, Y, test_size=0.5, random_state=randint)
        model.fit(X_1, y_1)
        outputs.append(model.predict(X_2))
        targets.append(y_2)

        model.fit(X_2, y_2)
        outputs.append(model.predict(X_1))
        targets.append(y_1)
    return outputs, targets


def _f_test_5x2(y_true, predictions_1, predictions_2, score_func):
    variances = []
    differences = []
    for i in range(5):
        index_1 = 2 * i
        index_2 = index_1 + 1

        score_diff_1 = score_func(y_true[index_1], predictions_1[index_1]) - score_func(y_true[index_1],
                                                                                        predictions_2[index_1])
        score_diff_2 = score_func(y_true[index_2], predictions_1[index_2]) - score_func(y_true[index_2],
                                                                                        predictions_2[index_2])

        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2

        differences.extend([score_diff_1 ** 2, score_diff_2 ** 2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    p_value = scipy.stats.f.sf(f_stat, 10, 5)

    return float(f_stat), float(p_value)


def multi_combined_ftest_5x2cv(models: List, X, Y, p_value=0.05, scoring=None, random_seed=None):
    scorer = get_scorer(scoring)
    score_func = partial(scorer._score_func, **scorer._kwargs)

    model_predictions = []
    all_targets = []
    rng = np.random.RandomState(random_seed)
    randints = [rng.randint(low=0, high=32767) for _ in range(5)]
    for model in models:
        output, target = _cv_5x2(model, X=X, Y=Y, randints=randints)
        model_predictions.append(output)
        all_targets.append(target)

    # check to make sure all folds had the same targets
    for pair in itertools.combinations([np.concatenate(target, axis=0) for target in all_targets], 2):
        assert np.array_equal(pair[0], pair[1])

    final_target = np.concatenate(all_targets[0], axis=0)

    chi2, cochrans_pvalue = cochrans_q(final_target,
                                       *[np.concatenate(predictions, axis=0) for predictions in model_predictions])
    log = {"Cochran's Q test:": (f"Q Chi^2: {chi2}", f"p-value: {cochrans_pvalue}")}

    if cochrans_pvalue <= p_value:
        # test each model
        for index_1, index_2 in itertools.combinations(range(len(model_predictions)), 2):
            # assert np.array_equal(all_targets[index_1], all_targets[index_2])
            log[f"{models[index_1]}-{models[index_2]}"] = _f_test_5x2(all_targets[index_1], model_predictions[index_1],
                                                                      model_predictions[index_2], score_func=score_func)
    return log
