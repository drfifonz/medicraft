"""
This module provides functionality for creating and managing pipelines in the medicraft project.

The module exports the following symbols:
- Pipeline: A class representing a pipeline.
- create_pipeline: A function for creating a pipeline.

Example usage:
    from pipeline import Pipeline, create_pipeline

    # Create a pipeline
    pipeline = create_pipeline()

    # Use the pipeline
    pipeline.run()
"""
__all__ = [
    "Pipeline",
]
from .pipeline import Pipeline
