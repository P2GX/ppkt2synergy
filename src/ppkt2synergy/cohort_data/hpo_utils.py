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

def calculate_pairwise_stats( 
        observed_status_A: pd.Series, 
        observed_status_B: pd.Series
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate statistical metrics including Spearman correlation, Kendall's Tau, and their p-values 
        for two binary (0/1) observed status vectors.

        Args:
            observed_status_A (pd.Series): Binary values (0/1) for the first variable. Must have at least 30 valid entries.
            observed_status_B (pd.Series): Binary values (0/1) for the second variable. Must have at least 30 valid entries.

        Returns:
            dict: A dictionary containing the following metrics:
                - Spearman: Spearman rank correlation coefficient.
                - Spearman_p_value: P-value for Spearman rank correlation tests.
                - Kendall's Tau: Kendall's Tau correlation coefficient.
                - Kendall_p_value: P-value for Kendall's Tau correlation tests.
        """

        results = {}  
        # Calculate Spearman correlation
        results["Spearman"], results["Spearman_p_value"] = scipy.stats.spearmanr(observed_status_A, observed_status_B)
        # Calculate Kendall's Tau
        results["Kendall"], results["Kendall_p_value"] = scipy.stats.kendalltau(observed_status_A, observed_status_B)
        # Phi coefficient
        confusion_matrix = pd.crosstab(observed_status_A, observed_status_B)
        try:
            chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
            n = confusion_matrix.sum().sum()
            phi = np.sqrt(chi2 / n)
            results["phi"], results["phi_p"] = phi, p
        except ValueError:
            results["phi"], results["phi_p"] = np.nan, np.nan

        return results