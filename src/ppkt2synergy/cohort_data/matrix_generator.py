import pandas as pd
import phenopackets as ppkt
from typing import List, Union, IO, Tuple, Callable, Optional
from .hpo_utils import load_hpo

class CohortMatrixGenerator:
    """
    Converts phenopacket data into structured matrices for analysis.

    This class generates:
    1. HPO Term Status Matrix: Indicates whether a patient has certain HPO terms.
    2. Disease Status Matrix: Indicates whether a patient has been diagnosed with specific diseases.

    Example:
        from ppkt2synergy import CohortDataLoader, CohortMatrixGenerator
        # Load phenopackets and create matrix generator
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> matrix_gen = CohortMatrixGenerator(phenopackets)
    """

    def __init__(
            self, 
            phenopackets: List[ppkt.Phenopacket], 
            hpo_file: Union[IO, str] = None,
            external_target_matrix: Optional[pd.DataFrame] = None):
        """
        Initializes the CohortMatrixGenerator.

        Args:
            phenopackets (List[Phenopacket]): List of phenopacket instances.
            hpo_file (str or IO, optional): Path to an HPO ontology file. Loads the latest version if not provided.

        Raises:
            ValueError: If the phenopackets list is empty.
            ValueError: If the HPO file fails to load.
        """
        if not phenopackets:
            raise ValueError("Phenopackets list cannot be empty.")

        self.phenopackets = phenopackets

        try:
            self.hpo = load_hpo(hpo_file)
        except (IOError, KeyError) as e:
            raise ValueError(f"Failed to load HPO file: {e}")
        self.hpo_term_observation_matrix, self.hpo_labels = self.generate_hpo_term_status_matrix()
        if external_target_matrix is not None:
            # Align rows to phenopacket IDs
            ppkt_ids = [ppkt.id for ppkt in self.phenopackets]
            self.target_matrix = external_target_matrix.reindex(ppkt_ids)
            self.target_labels = {
                col: col for col in self.target_matrix.columns} 
        else:
            self.target_matrix, self.target_labels = self.generate_target_status_matrix()

    def generate_hpo_term_status_matrix(
            self, 
            use_labels: bool = False,
            propagate_hierarchy: bool = True,
            ) -> Tuple[pd.DataFrame, dict]:
        """
        Creates a binary matrix indicating the presence/absence of HPO terms for each patient.

        Matrix structure:
        - Rows: Individual patient IDs.
        - Columns: Unique HPO terms (e.g., HP:0004322 = "Seizures").
        - Values:
            - 1 → Term is observed in the patient.
            - 0 → Term is explicitly excluded for the patient.
            - NaN → No information available.

        Example output (before propagation):
                     HP:0004322  HP:0001250  HP:0012759
        Patient_1         1         NaN         0
        Patient_2         NaN        1          NaN
     
        
        Propagation rules:
        - Observed terms (1) propagate to their ancestors.
        - Excluded terms (0) propagate to their descendants.

        Args:
            use_labels (bool): If True, replaces HPO term IDs with their human-readable labels.
            propagate_hierarchy (bool): If True, applies hierarchical propagation (only for HPO terms).

        Returns:
            Tuple:
                - pd.DataFrame: The HPO term status matrix.
                - dict: A mapping of HPO term IDs to their labels.
        """
        return self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.type.id, f.type.label, 0 if f.excluded else 1) for f in ppkt.phenotypic_features
            ],
            propagate_hierarchy=propagate_hierarchy,
            use_labels=use_labels
        )

    def generate_target_status_matrix(
            self, 
            use_labels: bool = False
            ) -> Tuple[pd.DataFrame, dict]:
        """
        Creates a binary matrix indicating the presence/absence of diagnosed diseases.

        Matrix structure:
        - Rows: Individual patient IDs.
        - Columns: Unique disease IDs (e.g., OMIM:101600 = "Marfan Syndrome").
        - Values:
            - 1 → Patient is diagnosed with this disease.
            - 0 → No diagnosis (default).
        
        Example output:
                     OMIM:101600  OMIM:603903
        Patient_1         1            0
        Patient_2         0            1

        Args:
            use_labels (bool): If True, replaces disease IDs with their human-readable labels.

        Returns:
            Tuple:
                - pd.DataFrame: The disease status matrix.
                - dict: A mapping of disease IDs to their labels.
        """
        return self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (d.disease.id, d.disease.label, 1) 
                for interp in (ppkt.interpretations or []) 
                if interp.diagnosis and interp.diagnosis.disease
                for d in [interp.diagnosis]
            ],
            propagate_hierarchy=False,
            use_labels=use_labels
        )

    def _generate_status_matrix(
        self, 
        feature_extractor: Callable, 
        propagate_hierarchy: bool, 
        use_labels: bool
    ) -> Tuple[pd.DataFrame, dict]:
        """
        General method to construct status matrices for either HPO terms or diseases.

        Args:
            feature_extractor (Callable): Function to extract features (HPO terms or diseases) from a phenopacket.
            propagate_hierarchy (bool): If True, applies hierarchical propagation (only for HPO terms).
            use_labels (bool): If True, replaces term/disease IDs with their human-readable labels.

        Returns:
            - pd.DataFrame: The resulting binary matrix.
            - dict: A mapping of feature IDs to their labels.
        """
        feature_ids, feature_labels, status_data = set(), {}, {}

        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            for f_id, f_label, value in feature_extractor(phenopacket):
                feature_ids.add(f_id)
                feature_labels[f_id] = f_label
                status_data[phenopacket.id][f_id] = value

        matrix = pd.DataFrame.from_dict(status_data, orient='index', columns=sorted(feature_ids))
        
        if propagate_hierarchy:
            matrix = self._propagate_hpo_hierarchy(matrix)
        
        if use_labels:
            matrix = matrix.rename(columns=feature_labels)

        return matrix, feature_labels

    def _propagate_hpo_hierarchy(
            self, 
            matrix: pd.DataFrame
            ) -> pd.DataFrame:
        """
        Propagates HPO term statuses using ontology hierarchy.

        Propagation rules:
        - Observed terms (1) propagate to their ancestors.
        - Excluded terms (0) propagate to their descendants.

        Example:
        HP:0012759 (Neurological abnormality)
        ├── HP:0001250 (Seizures)
        │   ├── HP:0004322 (Focal seizures)
        Before propagation:
                     HP:0004322  HP:0001250  HP:0012759
        Patient_1         1         NaN         NaN
        Patient_2         NaN        0          NaN

        After propagation:
                     HP:0004322  HP:0001250  HP:0012759
        Patient_1         1         1         1   # 1 propagates to ancestors
        Patient_2         0         0         NaN   # 0 propagates to descendants


        Args:
            matrix (pd.DataFrame): The initial HPO status matrix.

        Returns:
            `pd.DataFrame`: The matrix after propagation.
        """
        for hpo_id in matrix.columns:
            ancestors = {term.value for term in self.hpo.graph.get_ancestors(hpo_id)}
            descendants = {term.value for term in self.hpo.graph.get_descendants(hpo_id)}

            valid_ancestors = ancestors & set(matrix.columns)
            valid_descendants = descendants & set(matrix.columns)

            if valid_ancestors:
                mask = matrix[hpo_id] == 1
                matrix.loc[mask, list(valid_ancestors)] = 1  

            if valid_descendants:
                mask = matrix[hpo_id] == 0
                matrix.loc[mask, list(valid_descendants)] = 0  
        return matrix
