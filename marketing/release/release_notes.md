# FORGE v3.0.1 release notes

FORGE distills large vision-language-action teachers into compact students for edge robotics.
This launch ships the truthful CLI, mandatory teacher runtimes, trained-checkpoint provenance,
chunk-aware compression, ONNX/TensorRT/MLX export, and clean Python 3.12 packaging.

## Verified launch measurements

No launch measurements are published yet. Corrected-preprocessing training and
artifact validation must complete before this section gains any result rows.

The release kit deliberately carries no unvalidated performance claim.
Unfinished variants are deliberately omitted.

## Start

```sh
curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/anima-forge/main/install.sh | sh
forge doctor
forge quickstart --yes
```
