from typing import Union
import numpy as np
import phenopackets as ppkt
#from sklearn.metrics import matthews_corrcoef

def get_status_for_terms(
        patient: ppkt.Phenopacket, 
        hpo_id_A: str, 
        hpo_id_B: str) -> Union[tuple, None]:
    """
    Checks the statuses (observed or excluded) of two HPO term IDs in a patient's phenotype data.

    Args:
        phenotype: A PhenotypicFeature object containing the patient's phenotype data.
        hpo_id_A: The first HPO term ID to check.
        hpo_id_B: The second HPO term ID to check.

    Returns:
        tuple: A tuple (status_A, status_B) where each status is:
            - 1 if the phenotype is observed (not excluded).
            - 0 if the phenotype is excluded.
            - None if the HPO term is not present in the phenotype.
    """
    status_A = status_B = None

    for phenotype_feature in patient.phenotypic_features:
        if phenotype_feature.type.id == hpo_id_A:
            status_A = 0 if phenotype_feature.excluded else 1
        elif phenotype_feature.type.id == hpo_id_B:
            status_B = 0 if phenotype_feature.excluded else 1
    
        # Early stop if both statuses are found
        if status_A is not None and status_B is not None:
            return status_A, status_B    

    return None

