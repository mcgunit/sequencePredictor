import os, sys
import lightgbm as lgb

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


class LightGBMBackend(BoostingPredictorBase):
    """
    LightGBM-specific classifier construction. All shared logic is in
    BoostingBase, so this is a genuine like-for-like comparison against
    XGBoost/CatBoost - same features, same ticket construction, same subset
    generator, only the tree learner differs.

    LightGBM grows leaf-wise (rather than XGBoost's level-wise default), which
    fits a given number of trees faster but overfits small data more eagerly -
    hence min_child_samples and the explicit num_leaves cap below, both driven
    by the same tuned knobs the other backends use.
    """

    library_name = "LightGBM"

    def _make_classifier(self, num_classes):
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            # Leaf-wise growth ignores max_depth unless num_leaves is also
            # bounded: left at the default 31 a depth-2 model would still grow
            # 31 leaves, so the tuned depth would silently do nothing.
            num_leaves=max(2, 2 ** self.max_depth),
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            subsample_freq=1,  # subsample is ignored entirely without this
            colsample_bytree=self.colsample_bytree,
            # LightGBM's closest analogue to min_child_weight: the raw sample
            # count per leaf rather than a Hessian sum, so the tuned value maps
            # to a comparable "don't split on almost nothing" constraint.
            min_child_samples=max(1, int(self.min_child_weight)),
            reg_lambda=self.reg_lambda,
            n_jobs=self.num_threads,
            verbose=-1,  # otherwise every fit prints "No further splits with positive gain"
        )

        if num_classes > 2:
            params.update(objective="multiclass", num_class=num_classes)
        else:
            params.update(objective="binary")

        return lgb.LGBMClassifier(**params)


class LightGBMPredictor(LightGBMBackend, PerPositionBoostingPredictor):
    """LightGBM, one multiclass classifier per draw position ("LightGBM Model")."""


class LightGBMMultiLabelPredictor(LightGBMBackend, MultiLabelBoostingPredictor):
    """LightGBM, one binary classifier per number ("LightGBMMultiLabel Model")."""


if __name__ == "__main__":
    print("Trying LightGBM")

    name = 'vikinglotto'
    generateSubsets = [6, 7] if "keno" in name else []
    specialColumnCount = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}.get(name, 0)
    skipLastColumns = 1 if name == "lotto" else 0

    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    for label, model in [("per-position", LightGBMPredictor()), ("multi-label", LightGBMMultiLabelPredictor())]:
        model.setDataPath(dataPath)
        model.setSortedPrediction(not ("pick3" in name))
        model.setNumThreads(4)

        predicted_numbers, subsets = helpers.run_model_with_special_column(
            model, generateSubsets=generateSubsets,
            skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        print(f"{label}: {predicted_numbers}  subsets={subsets}")
