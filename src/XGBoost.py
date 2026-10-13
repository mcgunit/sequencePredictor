import os, sys
import numpy as np
import xgboost as xgb

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


class XGBoostBackend(BoostingPredictorBase):
    """XGBoost-specific classifier construction. All shared logic is in BoostingBase."""

    library_name = "XGBoost"

    def _make_classifier(self, num_classes):
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            # 1 by default: Backtester/HyperoptBoost already run many days (or
            # trials) across a process Pool, and letting each of those spawn
            # cpu_count() library threads oversubscribes the machine badly.
            # Callers running single-process raise this via setNumThreads.
            n_jobs=self.num_threads,
        )

        if num_classes > 2:
            params.update(objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss")
        else:
            params.update(objective="binary:logistic", eval_metric="logloss")

        return xgb.XGBClassifier(**params)

    def _fit_classifier(self, model, X, y, sample_weight=None):
        """
        Fits, with min_child_weight capped at what the history can carry (13
        Oct 2026). Under multi:softprob every class starts at p = 1/K, so each
        training row adds 2 x (1/K) x (1 - 1/K) to a node's Hessian sum, and a
        leaf whose sum stays under min_child_weight gets weight 0: with fewer
        rows than min_child_weight / (subsample x that) every tree is a single
        zero leaf and predict_proba is exactly 1/K. Vikinglotto's served 8.49
        needed about 396 rows of its 262 (306 draws): its XGBoost row scored
        every number 1/48 and played [1, 45, 46, 47, 48] for weeks, and short
        histories (the meta-learner tables' oldest days) hit the same wall.
        The cap is half the root's expected Hessian sum, so a first split is
        always possible; it never binds on the long games (lotto, keno,
        euromillions, eurodreams live). Padding rows carry a tiny sample
        weight, so the weight sum - not the row count - is what counts.
        """
        num_class = int(model.get_params().get("num_class") or 0)
        if num_class > 2:
            rows = float(np.sum(sample_weight)) if sample_weight is not None else float(len(y))
            cap = 0.5 * rows * float(self.subsample) * 2.0 * (1.0 / num_class) * (1.0 - 1.0 / num_class)
            if self.min_child_weight > cap:
                model.set_params(min_child_weight=cap)
        model.fit(X, y, sample_weight=sample_weight)
        return model


class XGBoostPredictor(XGBoostBackend, PerPositionBoostingPredictor):
    """
    XGBoost, one multiclass classifier per draw position.

    This is the original formulation and keeps its existing "XGBoost Model"
    row, so its tracked history stays continuous.
    """


class XGBoostMultiLabelPredictor(XGBoostBackend, MultiLabelBoostingPredictor):
    """
    XGBoost, one binary "is this number drawn" classifier per number.
    Tracked separately as "XGBoostMultiLabel Model" so the two formulations
    can be compared on identical data with the same library.
    """


# Backwards-compatible alias: the class was named after Keno back when it only
# ever ran there. Existing imports keep working.
XGBoostKenoPredictor = XGBoostPredictor


if __name__ == "__main__":
    print("Trying XGBoost")

    name = 'vikinglotto'
    generateSubsets = [6, 7] if "keno" in name else []
    specialColumnCount = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}.get(name, 0)
    skipLastColumns = 1 if name == "lotto" else 0

    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    for label, model in [("per-position", XGBoostPredictor()), ("multi-label", XGBoostMultiLabelPredictor())]:
        model.setDataPath(dataPath)
        model.setSortedPrediction(not ("pick3" in name))
        model.setNumThreads(4)

        predicted_numbers, subsets = helpers.run_model_with_special_column(
            model, generateSubsets=generateSubsets,
            skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        print(f"{label}: {predicted_numbers}  subsets={subsets}")
