"""Exception hierarchy for the thesis pipeline.

All project-specific errors derive from ThesisHydropowerError, which lets
callers catch the entire domain with a single except clause while still
being able to discriminate at a finer granularity when needed.

The hierarchy mirrors the pipeline stages:
  DataContractError   — malformed or inconsistent input data
  IngestionError      — file reading and pairing failures
  PreprocessingError  — signal calibration, filtering, and segmentation
  FeatureExtractionError — extractor failures
  ModelError          — training, loading, or inference failures
"""


class ThesisHydropowerError(Exception):
    """Base exception for all project-specific errors."""


class DataContractError(ThesisHydropowerError):
    """Data contract construction or validation failed."""


class ChannelCountError(DataContractError):
    """Unexpected number of channels in input data."""


class SampleRateError(DataContractError):
    """Invalid or inconsistent sample rates."""


class DataIntegrityError(DataContractError):
    """Data contains NaN, Inf, or inconsistent dimensions."""


class IngestionError(ThesisHydropowerError):
    """Errors raised while reading or pairing input files."""


class PreprocessingError(ThesisHydropowerError):
    """Errors raised during preprocessing."""


class FilterError(PreprocessingError):
    """Filter design or application failed."""


class FeatureExtractionError(ThesisHydropowerError):
    """Feature extraction failed for one or more channels."""


class ModelError(ThesisHydropowerError):
    """Base error for model training, export, and inference."""


class ModelTrainingError(ModelError):
    """Model training or artifact export failed."""


class ModelInferenceError(ModelError):
    """Model loading or inference execution failed."""


class ModelSchemaError(ModelInferenceError):
    """Feature schema mismatch between training artifacts and runtime data."""
