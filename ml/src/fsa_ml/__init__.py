"""fsa_ml — the ML platform.

    features/            feature definitions, point-in-time correct by construction
    fraud/               LightGBM training, calibration, SHAP, ONNX export   (M4)
    forecast/            hierarchical spend forecasting + naive baseline     (M5)
    receipt_extraction/  LoRA fine-tune of a small VLM (the GPU workload)    (M7)
    monitoring/          Evidently drift jobs -> Pushgateway                 (M5)
    registry/            MLflow helpers, champion/challenger promotion       (M4)

Layering rule (CLAUDE.md, enforced by import-linter): `ml/` may import `packages/*`
and nothing from `services/*`. Training must never require a running API.
"""
