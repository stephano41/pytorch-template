from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import ray
from mlxtend.evaluate import BootstrapOutOfBag
from mlxtend.evaluate.bootstrap_point632 import no_information_rate
from sklearn.metrics import check_scoring
from tqdm import tqdm


def bootstrap(model, X, Y, scoring, iters: int = 500, alpha: float = 0.95, num_cpu: int = 2, method: str = '.632',
              num_gpu=0):
    if method not in [".632", ".632+", "oob"]:
        raise ValueError(f"invalid bootstrap method {method}")
    scorer = check_scoring(model, scoring)
    score_func = partial(scorer._score_func, **scorer._kwargs)

    oob = BootstrapOutOfBag(n_splits=iters)

    if num_gpu <= 0:
        # use default python
        partial_bootstrap = partial(_one_bootstrap, model=model, scoring_func=score_func,
                                    X=X, Y=Y, method=method)
        with Pool(num_cpu, context="spawn") as pool:
            scores = []
            for score in tqdm(pool.imap_unordered(partial_bootstrap, oob.split(X)), total=oob.n_splits):
                scores.append(score)
    else:
        remote_bootstrap = ray.remote(num_gpus=num_gpu, max_calls=1)(_one_bootstrap)
        model_id, X_id, Y_id, score_func_id, method_id = ray.put(model), ray.put(X), ray.put(Y), ray.put(
            score_func), ray.put(method)
        scores = ray.get([remote_bootstrap.remote(idx, model=model_id, scoring_func=score_func_id,
                                                  X=X_id, Y=Y_id, method=method_id) for idx in
                          oob.split(X)])
        ray.shutdown()

    # scores = [ray.get(work.remote()) for _ in range(iters)]
    return get_ci(np.asarray(scores), alpha)


def _one_bootstrap(idx, model, scoring_func, X, Y, method='.632'):
    train_idx = idx[0]
    test_idx = idx[1]

    model.fit(X[train_idx], Y[train_idx])
    predicted_test_val = model.predict(X[test_idx])

    predicted_train_val = model.predict(X)
    test_acc = scoring_func(Y[test_idx], predicted_test_val)
    test_err = 1 - test_acc

    # training error on the whole training set as mentioned in the
    # previous comment above
    train_err = 1 - scoring_func(Y, predicted_train_val)
    if method == "oob":
        acc = test_acc
    else:
        if method == ".632+":
            gamma = 1 - (
                no_information_rate(Y, model.predict(X), scoring_func)
            )
            R = (test_err - train_err) / (gamma - train_err)
            weight = 0.632 / (1 - 0.368 * R)

        else:
            weight = 0.632

        acc = 1 - (weight * test_err + (1.0 - weight) * train_err)
    return acc


def get_ci(data, alpha=0.95):
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(data, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(data, p))
    return lower, upper
