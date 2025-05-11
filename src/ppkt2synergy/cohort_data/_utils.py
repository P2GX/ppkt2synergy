import hpotk
from typing import Union, IO, Dict
import pandas as pd
import scipy.stats
import numpy as np

def load_hpo(file: Union[IO, str] = None) -> hpotk.MinimalOntology:
    """Load HPO ontology from file or latest version
    
    Args:
        file: Path/File object or None for latest version
    
    Returns:
        MinimalOntology: Loaded HPO ontology
    """
    if file is None:
        store = hpotk.configure_ontology_store()
        return store.load_minimal_hpo()
    return hpotk.load_minimal_ontology(file)
