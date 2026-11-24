from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("numcompute")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "io",
    "preprocessing",
    "sort_search",
    "rank",
    "stats",
    "metrics",
    "optim",
    "pipeline",
    "utils",
    "benchmarking",
]