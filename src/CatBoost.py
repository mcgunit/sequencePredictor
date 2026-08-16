import os, sys
from catboost import CatBoostClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from BoostingBase import (
    BoostingPredictorBase,
    PerPositionBoostingPredictor,
    MultiLabelBoostingPredictor,
    helpers,
)


class CatBoostBackend(BoostingPredictorBase):
    """
    CatBoost-specific classifier construction. All shared logic is in
    BoostingBase, so this is a like-for-like comparison against
    XGBoost/LightGBM - same features, same ticket construction, same subset
    generator, only the tree learner differs.

    CatBoost builds symmetric (oblivious) trees, where every node at a given
    depth splits on the same feature. That's a strong built-in regulariser,
    which is the reason to have it here: on a few hundred draws it should
    overfit less eagerly than the other two, and if that matters, the tracked
    rows will show it.
    """

    library_name = "CatBoost"

    def _make_classifier(self, num_classes):
        params = dict(
            iterations=self.n_estimators,
            # CatBoost's symmetric trees make depth far more expensive than in
            # the other backends (a depth-d tree is always full, 2^d leaves),
            # and it rejects depth > 16 outright. Shared tuning ranges go to
            # 10, so this only ever clamps a deliberately extreme value.
            depth=min(max(1, self.max_depth), 16),
            learning_rate=self.learning_rate,
            # CatBoost's L2 term; same role as the other backends' reg_lambda.
            l2_leaf_reg=self.reg_lambda,
            min_data_in_leaf=max(1, int(self.min_child_weight)),
            rsm=self.colsample_bytree,
            thread_count=self.num_threads,
            # Every fit otherwise prints a full per-iteration training table -
            # unusable inside a Backtester pool running hundreds of fits.
            verbose=False,
            allow_writing_files=False,  # otherwise it litters catboost_info/ next to the cwd
        )

        if num_classes > 2:
            params.update(loss_function="MultiClass")
        else:
            params.update(loss_function="Logloss")

        # Bayesian bootstrap (CatBoost's default) rejects the subsample knob;
        # Bernoulli is the mode where it applies, matching the other backends.
        if self.subsample < 1.0:
            params.update(bootstrap_type="Bernoulli", subsample=self.subsample)

        return CatBoostClassifier(**params)


class CatBoostPredictor(CatBoostBackend, PerPositionBoostingPredictor):
    """CatBoost, one multiclass classifier per draw position ("CatBoost Model")."""


class CatBoostMultiLabelPredictor(CatBoostBackend, MultiLabelBoostingPredictor):
    """CatBoost, one binary classifier per number ("CatBoostMultiLabel Model")."""


if __name__ == "__main__":
    print("Trying CatBoost")

    name = 'vikinglotto'
    generateSubsets = [6, 7] if "keno" in name else []
    specialColumnCount = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}.get(name, 0)
    skipLastColumns = 1 if name == "lotto" else 0

    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    for label, model in [("per-position", CatBoostPredictor()), ("multi-label", CatBoostMultiLabelPredictor())]:
        model.setDataPath(dataPath)
        model.setSortedPrediction(not ("pick3" in name))
        model.setNumThreads(4)

        predicted_numbers, subsets = helpers.run_model_with_special_column(
            model, generateSubsets=generateSubsets,
            skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        print(f"{label}: {predicted_numbers}  subsets={subsets}")
