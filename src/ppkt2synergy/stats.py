import numpy as np
from ppkt2synergy import CohortManager
import pandas as pd
import hpotk
import typing
import scipy.stats
from sklearn.metrics import matthews_corrcoef

class CohortStats:
    """
    A class to analyze and compute statistical relationships between HPO terms 
    in a given cohort of patients. It generates HPO term observation matrices, 
    performs filtering, and calculates correlations between terms.
    """


    def __init__(
        self, 
        cohort_name: str, 
        ppkt_store_version: str,
        url= 'https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-10-09/hp.json'
    ):
        """
        Initialize the CohortStats class with a cohort name and Phenopackets store version.

        Args:
            cohort_name (str): The name of the patient cohort.
            ppkt_store_version (str): The version of the Phenopackets store.
            url: The URL or path to load the HPO ontology in JSON format.
        """
        
        self.phenopackets = CohortManager.from_ppkt_store(cohort_name,ppkt_store_version)
        self.hpo_term_observation_matrix = self.generate_hpo_term_status_matrix()
        self.hpo = hpotk.load_minimal_ontology(url)


    def generate_hpo_term_status_matrix(
        self
    ) -> pd.DataFrame:
        """
        Generate a binary matrix of HPO term statuses for all patients in a cohort.
        
        Returns:
            A pandas DataFrame:
                - Rows represent patients (indexed by their IDs).
                - Columns represent unique HPO IDs (sorted alphabetically).
                - Values are binary (1: observed, 0: excluded).
                - Missing values will remain as NaN.
        """
        
        hpo_ids = set()
        status_data = {}
        
        # Single pass to collect HPO IDs and patient statuses
        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            for feature in phenopacket.phenotypic_features:
                hpo_id = feature.type.id
                hpo_ids.add(hpo_id)
                status_data[phenopacket.id][hpo_id] = 0 if feature.excluded else 1

        # Create a sorted list of unique HPO IDs
        hpo_ids = sorted(hpo_ids)

        # Convert the status_data dictionary to a DataFrame
        status_matrix = pd.DataFrame.from_dict(status_data, orient='index', columns=hpo_ids)

        return status_matrix
    
    
    def filter_hpo_terms_and_dropna(
        self, 
        hpo_id_A: str, 
        hpo_id_B: str
    ) -> pd.DataFrame:
        """
        Filter the status matrix to only include two specific HPO terms and drop rows with NaN values.
        
        Args:
            hpo_id_A: The first HPO term ID to filter.
            hpo_id_B: The second HPO term ID to filter.
            
        Returns:
            A pandas DataFrame:
                - Rows represent patients (indexed by their IDs).
                - Columns represent the two specific HPO IDs.
                - Rows with NaN values are dropped.
        """

        # Check if the HPO terms exist in the matrix
        if hpo_id_A not in self.hpo_term_observation_matrix.columns or hpo_id_B not in self.hpo_term_observation_matrix.columns:
            raise ValueError(f"One or both of the HPO terms {hpo_id_A}, {hpo_id_B} are not present in the cohort.")

        filtered_matrix = self.hpo_term_observation_matrix[[hpo_id_A, hpo_id_B]]
        filtered_matrix_cleaned = filtered_matrix.dropna()
        return filtered_matrix_cleaned

        
    def find_unrelated_hpo_terms(
        self,
        hpo_id: str,  
    ) -> typing.List[str]:
        """
        Check if a given HPO ID  exists in the filtered terms_status DataFrame, where columns (HPO terms) with
        more than 50% missing values are removed. If it exists, return unrelated terms (terms not ancestors, 
        descendants, or the term itself). If the term is not found, notify the user.

        Args:
            term: The HPO ID to check.

        Returns:
            A list of unrelated HPO terms. If the HPO ID is not found in the filtered 
                          DataFrame, an error message is returned.
        """

        # Check if the term exists in the terms_status
        if hpo_id not in self.hpo_term_observation_matrix.columns:
            raise ValueError(f"The HPO term '{hpo_id}' is not found in this cohort.")
    
        # Filter out columns with more than 50% missing data
        filtered_matrix = self.hpo_term_observation_matrix.loc[:, self.hpo_term_observation_matrix.isna().mean() <= 0.5]

        if hpo_id not in filtered_matrix.columns:
            raise ValueError(f"error: The HPO term '{hpo_id}' was excluded due to missing data.")

        # Get ancestors and descendants of the term
        ancestors = self.hpo.graph.get_ancestors(hpo_id)
        descendants = self.hpo.graph.get_descendants(hpo_id)

        # Get unrelated terms
        related_terms = set(ancestors) | set(descendants) | {hpo_id}
        unrelated_terms = [col for col in filtered_matrix.columns if col not in related_terms]

        return unrelated_terms
    

    def calculate_stats(
        self, 
        observed_status_A: pd.Series, 
        observed_status_B: pd.Series
    ) -> typing.Dict[str, typing.Union[float, str]]:
        """
        Calculate Spearman, MCC, Fisher Exact, and their p-values for two observed statuses.

        Args:
            observed_status_A (pd.Series): Binary values (0/1) for the first variable.
            observed_status_B (pd.Series): Binary values (0/1) for the second variable.

        Returns:
            dict: A dictionary containing correlation coefficients and p-values for all metrics.
        """

        results = {}
        
        if len(observed_status_A) > 30:
            # Calculate Spearman correlation
            spearman_corr, spearman_p = scipy.stats.spearmanr(observed_status_A, observed_status_B)
            results["Spearman"] = spearman_corr
            results["Spearman_p_value"] = spearman_p

            # Construct the 2x2 contingency table
            n_11 = np.sum((observed_status_A == 1) & (observed_status_B == 1))
            n_10 = np.sum((observed_status_A == 1) & (observed_status_B == 0))
            n_01 = np.sum((observed_status_A == 0) & (observed_status_B == 1))
            n_00 = np.sum((observed_status_A == 0) & (observed_status_B == 0))
            contingency_table = np.array([[n_11, n_10], [n_01, n_00]])

            # Calculate MCC
            mcc = matthews_corrcoef(observed_status_A, observed_status_B)
            results["MCC"] = mcc
            chi2, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
            results["MCC_p_value"] = p_value
            
        elif len(observed_status_A)>= 5:
            odds_ratio, p_value = scipy.stats.fisher_exact(contingency_table)
            results["Fisher_exact"] = odds_ratio
            results["Fisher_Exact_p-value:"] = p_value

        else:
            results["Error"] = "Sample size is too small."

        return results


    
    def test_hpo_terms(
            self,               
            hpo_id_A: str, 
            hpo_id_B: str
        ) -> typing.Dict[str, typing.Union[float, str]]:
            """
            Perform correlation tests (Spearman, MCC, and phi) between two HPO terms in a cohort of patients.

            Args:
                hpo_id_A: The first HPO term ID to correlate.
                hpo_id_B: The second HPO term ID to correlate.

            Returns:
                dict: A dictionary containing correlation coefficients and p-values for each test.
            """

            # Check for ancestor/descendant relationship
            if self.hpo.graph.is_ancestor_of(hpo_id_A, hpo_id_B):
                raise ValueError(f"{hpo_id_A} is an ancestor of {hpo_id_B}")
            elif self.hpo.graph.is_descendant_of(hpo_id_A, hpo_id_B):
                raise ValueError(f"{hpo_id_A} is a descendant of {hpo_id_B}")
            
            try:
                observed_status = self.filter_hpo_terms_and_dropna(hpo_id_A, hpo_id_B)
            except ValueError as e:
                print(f"Error: {e}")

            # Perform all statistical tests
            stats = self.calculate_stats(observed_status[hpo_id_A], observed_status[hpo_id_B])
            return stats


    

