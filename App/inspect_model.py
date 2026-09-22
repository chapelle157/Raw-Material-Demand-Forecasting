from __future__ import annotations

import pickle
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / "hybrid_model.pkl"


def describe_model(obj, label: str) -> None:
    print(f"\n[{label}] type={type(obj)}")
    for attr in ["feature_names_in_", "feature_name_", "n_features_in_", "classes_"]:
        if hasattr(obj, attr):
            print(f"{attr}: {getattr(obj, attr)}")
    if hasattr(obj, "feature_name"):
        try:
            print(f"feature_name(): {obj.feature_name()}")
        except Exception as exc:  # noqa: BLE001
            print(f"feature_name() failed: {exc}")
    if hasattr(obj, "booster_") and hasattr(obj.booster_, "feature_name"):
        print(f"booster_.feature_name(): {obj.booster_.feature_name()}")


def main() -> None:
    print(f"Model path: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)
    print(f"Bundle type: {type(bundle)}")
    if isinstance(bundle, dict):
        print(f"Bundle keys: {list(bundle.keys())}")
        print(f"Threshold: {bundle.get('threshold')}")
        print("model_features:")
        for index, feature in enumerate(bundle.get("model_features", []), start=1):
            print(f"{index:03d}. {feature}")
        for key in ["clf_tuned", "cal_tuned", "reg_tuned"]:
            if key in bundle:
                describe_model(bundle[key], key)
    else:
        describe_model(bundle, "model")


if __name__ == "__main__":
    main()
