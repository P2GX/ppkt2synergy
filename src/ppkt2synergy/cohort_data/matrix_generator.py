import pandas as pd
import phenopackets as ppkt
from typing import List, Union, IO, Tuple, Callable, Optional
from ._utils import load_hpo
from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets
from gpsea.analysis.predicate import variant_effect
from gpsea.analysis.clf import monoallelic_classifier
from gpsea.model import VariantEffect
from gpsea.analysis.predicate import variant_effect, anyof

class PhenopacketMatrixGenerator:
    """
    Generates structured matrices from phenopacket data for downstream analysis.

    This class supports:
    1. HPO Term Status Matrix — capturing the presence or exclusion of HPO terms per patient.
    2. Disease Status Matrix — indicating diagnoses assigned to each patient.
    3. Optional target matrix — includes additional labels, such as variant effect classification.

    Example:
        from ppkt2synergy import CohortDataLoader, CohortMatrixGenerator
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> matrix_gen = PhenopacketMatrixGenerator(phenopackets)
    """

    def __init__(
            self, 
            phenopackets: List[ppkt.Phenopacket], 
            hpo_file: Union[IO, str] = None,
            use_labels: bool = False,
            variant_effect_type: Optional[VariantEffect] = None,
            mane_tx_id: Optional[Union[str, List[str]]] = None,
            external_target_matrix: Optional[pd.DataFrame] = None):
        """
        Args:
            phenopackets (List[Phenopacket]): A list of Phenopacket instances.
            hpo_file (str or IO, optional): Path to an HPO ontology file. Loads the latest version if None.
            variant_effect_type (Optional[str]): An optional variant effect type. Should be a member of `VariantEffect` Enum from `gpsea.model`, e.g., VariantEffect.MISSENSE_VARIANT.
            mane_tx_id (Optional[str or List[str]]): MANE transcript ID(s) used for variant effect analysis.
            external_target_matrix (Optional[pd.DataFrame]): Optional predefined target matrix to override defaults.

        Raises:
            ValueError: If `phenopackets` is empty or if HPO file fails to load.
        """
        if not phenopackets:
            raise ValueError("Phenopackets list cannot be empty.")
        
        self.phenopackets = phenopackets

        try:
            self.hpo = load_hpo(hpo_file)
        except (IOError, KeyError) as e:
            raise ValueError(f"Failed to load HPO file: {e}")
        
        self.hpo_term_observation_matrix, self.hpo_labels = self.generate_hpo_term_status_matrix(use_labels,propagate_hierarchy=True)
        self.hpo_term_observation_matrix = self.hpo_term_observation_matrix.reindex([ppkt.id for ppkt in self.phenopackets])
        base_target_matrix, base_target_labels = self.generate_target_status_matrix(use_labels)
        sex_matrix = self._generate_sex_matrix()
        base_target_matrix = pd.concat([base_target_matrix, sex_matrix], axis=1)
        base_target_labels = {**base_target_labels, "sex": "sex"}
        all_target_matrices = [base_target_matrix]
        all_target_labels = dict(base_target_labels)

        if variant_effect_type and mane_tx_id:
            label = str(variant_effect_type)

            cohort_creator = configure_caching_cohort_creator(self.hpo)
            cohort, _ = load_phenopackets(phenopackets=self.phenopackets, cohort_creator=cohort_creator)

            if isinstance(mane_tx_id, list):
                predicates = [variant_effect(variant_effect_type, tx_id=tx) for tx in mane_tx_id]
                predicate = anyof(predicates)
            else:
                predicate = variant_effect(variant_effect_type, tx_id=mane_tx_id)

            clf = monoallelic_classifier(
                a_predicate=predicate,
                b_predicate=~predicate,
                a_label=label,
                b_label="other"
            )

            variant_matrix = pd.DataFrame(
                data=[1 if (cat := clf.test(p)) and cat.category.name == label else 0 for p in cohort],
                index=[p.labels._meta_label for p in cohort],
                columns=[label]
            )

            all_target_matrices.append(variant_matrix)
            all_target_labels[label] = label

        if external_target_matrix is not None:
            if not isinstance(external_target_matrix, pd.DataFrame):
                raise ValueError("external_target_matrix must be a pandas DataFrame")
            ppkt_ids = [ppkt.id for ppkt in self.phenopackets]
            ext_matrix = external_target_matrix.reindex(ppkt_ids)
            all_target_matrices.append(ext_matrix)
            all_target_labels.update({col: col for col in ext_matrix.columns})
            
        self.target_matrix = pd.concat(all_target_matrices, axis=1).reindex([ppkt.id for ppkt in self.phenopackets])
        self.target_labels = all_target_labels
        


            

    def generate_hpo_term_status_matrix(
            self, 
            use_labels: bool = False,
            propagate_hierarchy: bool = True,
            ) -> Tuple[pd.DataFrame, dict]:
        """
        Constructs a binary matrix indicating the presence or exclusion of HPO terms for each patient.

        Structure of the resulting matrix:
        - Rows: Patient IDs
        - Columns: HPO term IDs (e.g., HP:0004322)
        - Values:
            - 1 → Term is observed in the patient
            - 0 → Term is explicitly excluded
            - NaN → No information

        Propagation (if enabled):
        - Observed terms (1) propagate to all ancestors in the ontology
        - Excluded terms (0) propagate to all descendants

        Args:
            use_labels (bool): If True, replaces HPO IDs with human-readable labels.
            propagate_hierarchy (bool): If True, applies ontology-based propagation.

        Returns:
            Tuple[pd.DataFrame, dict]: 
                - The HPO status matrix
                - A mapping from HPO term IDs to their labels
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
        Constructs a binary matrix indicating whether each patient has been diagnosed with specific diseases.

        Structure of the resulting matrix:
        - Rows: Patient IDs
        - Columns: Disease IDs (e.g., OMIM:101600)
        - Values:
            - 1 → Patient has been diagnosed with this disease
            - 0 → No diagnosis recorded (default)

        Args:
            use_labels (bool): If True, replaces disease IDs with their corresponding labels.

        Returns:
            Tuple[pd.DataFrame, dict]: 
                - The disease status matrix
                - A mapping from disease IDs to their labels
        """
        return self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.term.id, f.term.label, 0 if f.excluded else 1) for f in ppkt.diseases
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
        Internal method to generate a binary matrix from any phenopacket feature set (e.g., HPO terms, diseases).

        Args:
            feature_extractor (Callable): Function that extracts (id, label, value) tuples from each phenopacket.
            propagate_hierarchy (bool): If True, applies hierarchical propagation (only relevant for HPO).
            use_labels (bool): If True, replaces feature IDs with human-readable labels.

        Returns:
            Tuple[pd.DataFrame, dict]: 
                - The binary matrix
                - A mapping from feature IDs to their labels
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
        Applies hierarchical propagation to an HPO term matrix.

        Propagation logic:
        - A value of 1 (observed) propagates to all ancestors of the corresponding HPO term.
        - A value of 0 (excluded) propagates to all descendants.

        Example (before propagation):
            HP:0012759 (Neurological abnormality)
            ├── HP:0001250 (Seizures)
            │   └── HP:0004322 (Focal seizures)

            Matrix:
                         HP:0004322  HP:0001250  HP:0012759
            Patient_1         1         NaN         NaN
            Patient_2         NaN        0          NaN

        Example (after propagation):
                         HP:0004322  HP:0001250  HP:0012759
            Patient_1         1         1          1
            Patient_2         0         0          NaN

        Args:
            matrix (pd.DataFrame): HPO status matrix to propagate.

        Returns:
            pd.DataFrame: The matrix with propagated values.
        """
        invalid_terms = []
        for hpo_id in matrix.columns:
            try:
                ancestors = {term.value for term in self.hpo.graph.get_ancestors(hpo_id)}
                descendants = {term.value for term in self.hpo.graph.get_descendants(hpo_id)}

            except ValueError:
                invalid_terms.append(hpo_id)
                continue        

            valid_ancestors = ancestors & set(matrix.columns)
            valid_descendants = descendants & set(matrix.columns)

            if valid_ancestors:
                mask = matrix[hpo_id] == 1
                matrix.loc[mask, list(valid_ancestors)] = 1  

            if valid_descendants:
                mask = matrix[hpo_id] == 0
                matrix.loc[mask, list(valid_descendants)] = 0  
        if invalid_terms:
            matrix.drop(columns=invalid_terms, inplace=True) 
        return matrix
    
    def _generate_sex_matrix(self) -> pd.DataFrame:
        """
        Creates a binary sex matrix with:
        - 1 → Male
        - 0 → Female
        - NaN → Unknown, Other, or missing

        Returns:
            pd.DataFrame: Sex matrix with patient IDs as index and a single column 'male'.
        """
        sex_data = {}

        for ppkt in self.phenopackets:
            if ppkt.subject.sex == 'MALE':
                sex_data[ppkt.id] = 1
            elif ppkt.subject.sex == 'FEMALE':
                sex_data[ppkt.id] = 0
            else:
                sex_data[ppkt.id] = float('nan')

        return pd.DataFrame.from_dict(sex_data, orient='index', columns=['male'])

