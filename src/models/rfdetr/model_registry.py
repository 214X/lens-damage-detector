from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall


# RF-DETR package exposes Nano / Small / Medium.
# We map "low" to the Small variant so the training pipeline can use
# the user-facing naming nano -> low -> medium consistently.
MODEL_SEQUENCE = [
    ("nano", RFDETRNano),
    ("low", RFDETRSmall),
    ("medium", RFDETRMedium),
]
