"""
UI Pages Package

Exposes all available pages for the Bolsover District Council application.
"""

from .data_entry import data_entry_page
from .results import results_page
from .aggregated import aggregated_analysis_page

__all__ = [
    'data_entry_page',
    'results_page',
    'aggregated_analysis_page'
]