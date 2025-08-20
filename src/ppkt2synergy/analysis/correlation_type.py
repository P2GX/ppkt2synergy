from enum import Enum

class CorrelationType(Enum):
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PHI = "phi"