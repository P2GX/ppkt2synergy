import pandas as pd
import phenopackets as ppkt
from typing import List, Union, IO,Tuple
from .hpo_utils import load_hpo

class CohortMatrixGenerator:
    """
    Generates data matrices (HPO terms, diseases) from phenopacket data.

     Example:
        ```python
        from ppkt2synergy import CohortDataLoader
        from ppkt2synergy import CohortMatrixGenerator
        dataloader = CohortDataLoader.from_ppkt_store('FBN1')
        matrix_generator = CohortMatrixGenerator(dataloader)
        ```
    """

    def __init__(
        self, 
        phenopackets: List[ppkt.Phenopacket], 
        hpo_file: Union[IO, str] = None):
        """
        Initialize the CohortMatrixGenerator class with phenopacket data.

        Args:
            phenopackets: List of Phenopacket instances.
            hpo_file: Path to HPO file (optional, loads latest if None).

        Raises:
            ValueError: If phenopackets is empty or HPO loading fails.
            TypeError: If phenopackets contains invalid objects.
        """

        if not phenopackets:
            raise ValueError("Phenopackets list cannot be empty.")
        if not all(isinstance(p, ppkt.Phenopacket) for p in phenopackets):
            raise TypeError("All elements in phenopackets must be Phenopacket instances.")
        
        self.phenopackets = phenopackets

        try:
            self.hpo = load_hpo(hpo_file)
        except Exception as e:
            raise ValueError(f"Error loading HPO file: {e}")

        self.hpo_term_observation_matrix, self.hpo_labels = self.generate_hpo_term_status_matrix()
        self.disease_matrix, self.disease_labels = self.generate_disease_status_matrix()


    def generate_hpo_term_status_matrix(
        self,
        use_labels: bool = False
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generates a binary matrix for HPO term presence/absence.
        
        - Rows: Patient IDs
        - Columns: Unique HPO terms
        - Values:
            - 1 = Observed
            - 0 = Excluded
            - NaN = No data
        
        Propagation rules:
            - Observed (1) propagates to ancestors.
            - Excluded (0) propagates to descendants.
        
        Returns:
            A pandas DataFrame representing the HPO term status matrix.
            A dictionary mapping HPO IDs to their labels.
        """
        hpo_ids = set()
        status_data = {}
        hpo_labels = {}
        
        # Populate status data by iterating through the phenopackets
        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            for feature in phenopacket.phenotypic_features:
                hpo_id = feature.type.id
                hpo_label = feature.type.label
                hpo_ids.add(hpo_id)
                hpo_labels[hpo_id] = hpo_label
                status_data[phenopacket.id][hpo_id] = 0 if feature.excluded else 1

        # Convert the status_data dictionary to a DataFrame
        status_matrix = pd.DataFrame.from_dict(status_data, orient='index', columns=sorted(hpo_ids))
        hpo_matrix = self._propagate_hpo_hierarchy(status_matrix)
        if use_labels:
            status_matrix = status_matrix.rename(columns=hpo_labels)
        
        return hpo_matrix, hpo_labels
    
    def _propagate_hpo_hierarchy(
            self, 
            matrix: pd.DataFrame
            ) -> pd.DataFrame:
        """
        Propagates HPO term statuses using ontology hierarchy.
        
        - Observed (1) propagates to ancestors.
        - Excluded (0) propagates to descendants.
        """

        for hpo_id in matrix.columns:
            ancestors = {term.value for term in self.hpo.graph.get_ancestors(hpo_id)}
            descendants = {term.value for term in self.hpo.graph.get_descendants(hpo_id)}
            
            # Propagate observed (1) to ancestors
            valid_ancestors = ancestors & set(matrix.columns)
            if valid_ancestors:
                mask = matrix[hpo_id] == 1
                matrix.loc[mask, list(valid_ancestors)] = matrix.loc[mask, list(valid_ancestors)].fillna(1)
            
            # Propagate excluded (0) to descendants
            valid_descendants = descendants & set(matrix.columns)
            if valid_descendants:
                mask = matrix[hpo_id] == 0
                matrix.loc[mask, list(valid_descendants)] = matrix.loc[mask, list(valid_descendants)].fillna(0)
        
        return matrix

    def generate_disease_status_matrix(
            self,
            use_labels: bool = False
            ) -> Tuple[pd.DataFrame, dict]:
        """
        Generates a binary matrix for disease presence/absence.
        
        - Rows: Patient IDs
        - Columns: Unique disease IDs
        - Values:
            - 1 = Diagnosed
            - 0 = Not diagnosed (default)
        
        Returns:
            A pandas DataFrame representing the disease status matrix.
            A dictionary mapping disease IDs to their labels.
        """
        disease_ids = set()
        status_data = {}
        disease_labels = {}

        # Populate status data by iterating through the phenopackets
        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            if phenopacket.interpretations is not None and len(phenopacket.interpretations) > 0:
                for interpretation in phenopacket.interpretations:
                    diagnosis = interpretation.diagnosis
                    if diagnosis is not None and diagnosis.disease is not None:
                        disease_id = diagnosis.disease.id
                        disease_label = diagnosis.disease.label
                        disease_ids.add(disease_id)
                        disease_labels[disease_id] = disease_label
                        status_data[phenopacket.id][disease_id] = 1  # Diagnosed with the disease

        # Convert the status_data dictionary to a DataFrame
        disease_matrix = pd.DataFrame.from_dict(status_data, orient='index', columns=sorted(disease_ids))
        if use_labels:
            disease_matrix = disease_matrix.rename(columns= disease_labels)
        
        # Fill NaN values with 0 (no diagnosis = no disease)
        disease_matrix = disease_matrix.fillna(0)
        return disease_matrix, disease_labels
    
