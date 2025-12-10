"""cosmo_slam: Centralized SLAM backend for COSMO-bench JRL datasets.

This package provides:
- Robust JRL parsing with schema validation
- Data models for factors and initialization
- A global factor graph builder
- An iSAM2 manager for incremental optimization
- Robust noise builders (Huber/Cauchy)
- Basic visualization utilities
- A CLI entry point (see main.py)

Design intent:
Keep modules small and single-purpose so they can be swapped out
(e.g., different loaders or iSAM2 parameterizations) while the rest
of the pipeline remains stable.
"""
__all__ = ["loader", "models", "graph", "isam", "robust", "viz"]
__version__ = "0.1.0"
