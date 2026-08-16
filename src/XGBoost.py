import os, sys
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
