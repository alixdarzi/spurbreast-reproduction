# Traceability

| Implementation item | Paper evidence | Official repository evidence | Planned implementation | Status |
|---|---|---|---|---|
| Cohort | §2 Data | README | Official field-strength archive | Confirmed |
| Unit and label | §2, Fig. 1 | `training/{0,1}` etc. | Slice-level binary label | Confirmed |
| Buffer exclusion | §2, Fig. 1 | None | Trust released PNG membership | Rule unknown |
| Patient split | §3.1, Table 1 | Filename patient ID | Exact folder membership plus assertions | Confirmed |
| Shortcut direction | §4 | Not exposed | Join TCIA metadata | Confirmed |
| Image decoding | Not stated | PIL RGB conversion | PIL RGB conversion | High confidence |
| Train transform | Not stated | Resize 256, RandomResizedCrop 224, ToTensor | Match repository | Medium confidence |
| Evaluation transform | Not stated | Resize 224, ToTensor | Match repository | Medium confidence |
| Normalization | Not stated | None shown | None primary; ImageNet sensitivity | Assumption |
| Architecture | §3.3 | No model code | Torchvision ResNet-50, two logits | High confidence |
| ImageNet weights | §3.3 | Exact version absent | `IMAGENET1K_V1` | Assumption |
| Fine-tuning | Not stated | None | All layers trainable | Assumption |
| Loss | Not stated | None | Unweighted cross-entropy | Assumption |
| Optimizer search | §4 grid | None | Bounded validation-only plan | Partial reproduction |
| Batch and epochs | §4 | None | 32 and 50 | Confirmed |
| Seed | Not stated | None | 2025, 2026, 2027 | Assumption |
| Checkpoint | §4 validation accuracy | None | Accuracy; NLL/epoch tie-break | Partly assumed |
| Threshold | Not stated | None | Two-logit argmax | Assumption |
| Metrics | Table 2 | None | Global confusion matrix | Paper aggregation unresolved |
| Statistical analysis | None | None | Patient-cluster bootstrap | Added rigor |
