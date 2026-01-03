"""
Benchmark data modules for California welfare analysis.

This module provides interfaces for fetching and managing benchmark data
from official sources for validation and calibration:

- DOF population estimates
- Unauthorized immigrant population benchmarks (Pew, MPI, CMS, DHS)
"""

from .dof_population import DOFPopulationFetcher, get_ca_population, PopulationEstimate
from .unauthorized import UnauthorizedBenchmarkFetcher, UnauthorizedEstimate

__all__ = [
    "DOFPopulationFetcher",
    "get_ca_population",
    "PopulationEstimate",
    "UnauthorizedBenchmarkFetcher",
    "UnauthorizedEstimate",
]
