"""Paired held-out comparisons for the corrected SMOTE/ADASYN RFE runs."""

from __future__ import annotations

import json

import pandas as pd

from scripts.proposed_model import _delong_pvalue, _mcnemar_pvalue
from scripts.run_experiments import PAPER_PATHS, RESULTS, TARGET, _fit_phase1_model
from scripts.hard_timeout import ExperimentTimeout, run_with_hard_timeout


def run() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    runs = pd.read_csv(RESULTS / "all_runs.csv")
    records = []
    for sampler in ("SMOTE", "ADASYN"):
        stack_row = runs.query(
            "run_key == @key and status == 'ok'", local_dict={
                "key": f"phase1:paper_faithful:{sampler}:RFE:PaRSEL:42:rfe_fixedn_v3"
            }
        ).iloc[-1]
        selected_count = int(stack_row.rfe_features)
        stack_predictions = pd.read_csv(RESULTS / f"curve_{stack_row.run_id}.csv")
        timed_out = False
        for model_name in ("PaRSEL", "LogitBoost_surrogate"):
            if model_name == "PaRSEL":
                prediction = stack_predictions.prediction.to_numpy()
                scores = stack_predictions.score.to_numpy()
            else:
                prediction_path = RESULTS / f"phase1_{sampler.lower()}_rfe_surrogate_predictions.csv"
                if prediction_path.exists():
                    cached = pd.read_csv(prediction_path)
                    prediction, scores = cached.prediction.to_numpy(), cached.score.to_numpy()
                else:
                    try:
                        prediction, scores, *_ = run_with_hard_timeout(
                            _fit_phase1_model, train, test, model_name, "RFE", sampler,
                            False, selected_count, timeout_seconds=600,
                        )
                    except ExperimentTimeout as error:
                        metadata = json.loads(stack_row.reducer_metadata)
                        records.append({
                            "sampler": sampler,
                            "selected_feature_count": selected_count,
                            "selected_features": ";".join(metadata.get("selected_features", [])),
                            "comparison_status": "skipped (time-boxed)",
                            "error": f"timeout: {error}", "mcnemar_p": "", "delong_p": "",
                        })
                        timed_out = True
                        break
                    pd.DataFrame({"target": test[TARGET].to_numpy(), "prediction": prediction, "score": scores}).to_csv(
                        prediction_path, index=False
                    )
            if model_name == "PaRSEL":
                stack_prediction, stack_scores = prediction, scores
            else:
                surrogate_prediction, surrogate_scores = prediction, scores
        metadata = json.loads(stack_row.reducer_metadata)
        if timed_out:
            continue
        records.append({
            "sampler": sampler, "selected_feature_count": selected_count,
            "selected_features": ";".join(metadata.get("selected_features", [])),
            "comparison_status": "ok", "error": "",
            "ros_rfe_retained_all_features": "",
            "mcnemar_p": _mcnemar_pvalue(test[TARGET].to_numpy(), stack_prediction, surrogate_prediction),
            "delong_p": _delong_pvalue(test[TARGET].to_numpy(), stack_scores, surrogate_scores),
        })
    ros_row = runs.loc[
        runs.run_key.eq("phase1:paper_faithful:ROS:RFE:PaRSEL:42:rfe_fixedn_v3")
        & runs.status.eq("ok")
    ].iloc[-1]
    ros_metadata = json.loads(ros_row.reducer_metadata)
    records.append({
        "sampler": "ROS", "selected_feature_count": int(ros_row.rfe_features),
        "selected_features": ";".join(ros_metadata.get("selected_features", [])),
        "ros_rfe_retained_all_features": int(ros_row.rfe_features) == len(ros_metadata.get("selected_features", [])) == 8,
        "mcnemar_p": "", "delong_p": "",
    })
    pd.DataFrame(records).to_csv(RESULTS / "phase1_rfe_paired_tests.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    run()
