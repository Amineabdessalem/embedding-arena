"""embedding-arena: measure the recall and memory cost of compressed embeddings."""

from arena.compress import Compressor, binary, float32, int8, pca, truncate
from arena.evaluate import Report, evaluate

__all__ = ["Compressor", "Report", "binary", "evaluate", "float32", "int8", "pca", "truncate"]
__version__ = "0.1.0"
