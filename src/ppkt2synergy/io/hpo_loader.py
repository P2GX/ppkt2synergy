import hpotk
from typing import Union, IO,Optional

def load_hpo(
        file: Optional[Union[IO, str]] = None
    ) -> hpotk.MinimalOntology:
    """
    Loads the HPO ontology from a file or fetches the latest version.

    Args:
        file (Union[IO, str], optional): (default: None)
            Path or file object to the HPO ontology. If `None`, the latest version is loaded. 

    Returns:
        hpotk.MinimalOntology: 
            The loaded HPO ontology object.

    Example:
        # Load the latest version of the HPO ontology
        ontology = load_hpo()

        # Load from a file
        ontology = load_hpo('path/to/hpo_ontology.owl')
    """
    
    if file is None:
        store = hpotk.configure_ontology_store()
        return store.load_minimal_hpo()
    return hpotk.load_minimal_ontology(file)
