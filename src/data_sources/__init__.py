"""
Administrative data source downloaders for California welfare analysis.

This module provides standardized interfaces for fetching and cleaning
administrative data from various California state agencies and federal sources.

Modules:
    base: Abstract base classes and generic downloaders
    calfresh: CalFresh DFA256 participation data
    calworks: CalWORKs program data
    medi_cal: Medi-Cal administrative data
    ssi: SSA SSI recipient data
    registry: Data source registry and orchestration
"""

from .base import (
    BaseDataSourceDownloader,
    CkanDownloader,
    PortalDownloader,
    DataSourceMetadata,
    DownloadResult,
    DownloadOutcome,
)
from .calfresh import CalFreshDownloader
from .calworks import CalWORKsDownloader
from .medi_cal import MediCalDownloader
from .ssi import SSIDownloader
from .registry import AdminDataSourceRegistry

__all__ = [
    # Base classes
    "BaseDataSourceDownloader",
    "CkanDownloader",
    "PortalDownloader",
    "DataSourceMetadata",
    "DownloadResult",
    "DownloadOutcome",
    # Specific downloaders
    "CalFreshDownloader",
    "CalWORKsDownloader",
    "MediCalDownloader",
    "SSIDownloader",
    # Registry
    "AdminDataSourceRegistry",
]
