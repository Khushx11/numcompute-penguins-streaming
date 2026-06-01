from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("numcompute-stream")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "stats",
    "preprocessing",
    "metrics",
    "io",
    "pipeline",
    "tree",
    "ensemble",
    "stream",
    "visualise",
]
