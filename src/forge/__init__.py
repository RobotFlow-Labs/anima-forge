"""FORGE — VLA Model Distillation Pipeline for Edge Robotics.

Takes any 7B+ Vision-Language-Action model and compresses to <2GB
for real-time edge deployment on Jetson (TensorRT) and Apple Silicon (MLX).

Pipeline: Teacher Labels → Knowledge Distillation → Compression → Runtime Export
"""

__version__ = "3.0.1"
