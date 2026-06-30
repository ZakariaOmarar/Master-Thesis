"""Configuration constants for the sensor setup."""

from .constants import (
    ACCEL_COUNT,
    ACCEL_SAMPLE_RATE_TARGET,
    GENERATOR_LEVEL_Z_M,
    MIC_COUNT,
    MIC_SAMPLE_RATE,
    SENSOR_LAYOUT,
    TURBINE_LEVEL_Z_M,
)
from .device import describe_device, resolve_device

__all__ = [
    "ACCEL_COUNT",
    "ACCEL_SAMPLE_RATE_TARGET",
    "GENERATOR_LEVEL_Z_M",
    "MIC_COUNT",
    "MIC_SAMPLE_RATE",
    "SENSOR_LAYOUT",
    "TURBINE_LEVEL_Z_M",
    "describe_device",
    "resolve_device",
]
