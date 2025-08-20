import pandas as pd
import phenopackets as ppkt
from typing import List, Union, IO, Tuple, Callable, Optional, Iterable,Dict
from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets
from gpsea.analysis.clf import monoallelic_classifier
from gpsea.model import VariantEffect
from gpsea.analysis.predicate import variant_effect, anyof
from ._utils import HPOHierarchyUtils
import logging
logger = logging.getLogger(__name__)

class PhenopacketMatrixGenerator:
    """
    Generates structured matrices from phenopacket data for downstream analysis.

    This class supports generation of:
    1. **HPO Term Observation Matrix** — Indicates the presence or exclusion of HPO terms for each patient.
    2. **Disease Status Matrix** — Indicates diagnoses assigned to each patient.
    3. **Optional Target Matrix** — Includes additional labels (e.g., variant effect classifications).

    Example:
        from ppkt2synergy import CohortDataLoader, PhenopacketMatrixGenerator

        >>> phenopackets = CohortDataLoader.from_ppkt_store("FBN1")
        >>> matrix_gen = PhenopacketMatrixGenerator(phenopackets)
    """

    def __init__(
            self, 
            phenopackets: List[ppkt.Phenopacket], 
            hpo_file: Optional[Union[IO, str]] = None,
            variant_effect_type: Optional[VariantEffect] = None,
            mane_tx_id: Optional[Union[str, List[str]]] = None,
            external_target_matrix: Optional[pd.DataFrame] = None,
            hpo_hierarchy: Optional[HPOHierarchyUtils] = None
        ):
        """
        Args:
            phenopackets (List[Phenopacket]): 
                A list of Phenopacket instances.
            hpo_file (str or IO, optional): (default: None)
                Path to an HPO ontology file. Loads the latest version if None. 
            variant_effect_type (Optional[str]): (default: None)
                The specific variant effect class to evaluate.
                This should be a member of the `VariantEffect` Enum from `gpsea.model`, for example, `VariantEffect.MISSENSE_VARIANT` or `VariantEffect.NONSENSE_VARIANT`. 
            mane_tx_id (Optional[str or List[str]]): 
                MANE transcript ID(s) used to filter variant effects. (default: None)
                Different cohorts may use different transcripts, so providing one or more IDs ensures consistent variant effect classification across datasets. 
            external_target_matrix (Optional[pd.DataFrame]): (default: None)
                User-provided binary matrix (indexed by patient ID). 
                Columns with only valid values (0, 1, or NaN) are merged into the final target matrix; others are skipped with a warning.
            hpo_hierarchy (Optional[HPOHierarchyUtils]): (default: None)
                An instance of HPOHierarchyUtils for hierarchical processing.
                If None, a new instance is created using the provided `hpo_file`. 
        Raises:
            ValueError: If `phenopackets` is empty or if HPO file fails to load.
        """
        if not phenopackets:
            raise ValueError("Phenopackets list cannot be empty.")
        
        self.phenopackets = phenopackets
        self.variant_effect_type = variant_effect_type
        self.mane_tx_id = mane_tx_id
        self.external_target_matrix = external_target_matrix
        self.patient_index = pd.Index([p.id for p in phenopackets], name="patient_id")
        if not isinstance(hpo_hierarchy, HPOHierarchyUtils):
            self.hpo_hierarchy = HPOHierarchyUtils(hpo_file=hpo_file)
        else:
            self.hpo_hierarchy = hpo_hierarchy
        self.hpo_term_observation_matrix = self._generate_hpo_term_status_matrix()
        self.target_matrix, self.target_labels = self._compute_target()


    def _generate_hpo_term_status_matrix(
            self, 
            propagate_hierarchy: bool = True,
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
            propagate_hierarchy (bool): (default: True)
                If True, applies hierarchical propagation. 

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: 
                - Binary matrix of HPO term statuses
                - A mapping from HPO term IDs to their labels
        """
        return self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.type.id, 0 if f.excluded else 1) for f in ppkt.phenotypic_features
            ],
            propagate_hierarchy=propagate_hierarchy,
        )


    def _generate_disease_status_matrix(
            self, 
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Constructs a binary matrix indicating whether each patient has been diagnosed with specific diseases.

        Structure of the resulting matrix:
        - Rows: Patient IDs
        - Columns: Disease IDs (e.g., OMIM:101600)
        - Values:
            - 1 → Patient has been diagnosed with this disease
            - 0 → No diagnosis recorded (default)

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: 
                - Binary matrix of disease statuses
                - A mapping from disease IDs to their labels
        """
        return self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.term.id, 0 if f.excluded else 1) for f in ppkt.diseases
            ],
            propagate_hierarchy=False,
        )


    def _generate_status_matrix(
            self, 
            feature_extractor: Callable, 
            propagate_hierarchy: bool, 
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Internal method to generate a binary matrix from any phenopacket feature set (e.g., HPO terms, diseases).

        Args:
            feature_extractor (Callable): 
                Function that extracts (id, label, value) tuples from each phenopacket.
            propagate_hierarchy (bool): 
                If True, applies hierarchical propagation (only relevant for HPO).

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: 
                - Binary status matrix
                - A mapping from feature IDs to their labels
        """
        feature_ids, status_data = set(), {}

        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            for f_id, value in feature_extractor(phenopacket):
                feature_ids.add(f_id)
                status_data[phenopacket.id][f_id] = value

        matrix = pd.DataFrame.from_dict(
            status_data, 
            orient='index', 
            columns=sorted(feature_ids)
            ).reindex(self.patient_index)
        
        if propagate_hierarchy:
            matrix = self.hpo_hierarchy.propagate_hpo_hierarchy(matrix)

        return matrix

    
    def _generate_sex_matrix(
            self
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Creates a binary sex matrix with:

        Values:
        - 1 → Male
        - 0 → Female
        - NaN → Unknown, Other, or missing

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]:
                - Matrix with one column: 'sex'.
                - Label mapping for column.
        """
        sex_data = {}

        for ppkt in self.phenopackets:
            if ppkt.subject.sex == 'MALE':
                sex_data[ppkt.id] = 1
            elif ppkt.subject.sex == 'FEMALE':
                sex_data[ppkt.id] = 0
            else:
                sex_data[ppkt.id] = float('nan')

        sex_matrix = pd.DataFrame.from_dict(sex_data, orient='index', columns=['sex']).reindex(self.patient_index)
        return sex_matrix, {'sex': 'sex'}
    

    def _process_variant_effects(
            self,
            variant_effect_type: VariantEffect,
            mane_tx_id: Union[str, List[str]]
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:

        """
        Processes variant effect annotations and creates a classification matrix.

        If both `variant_effect_type` and `mane_tx_id` are provided, assigns:
        - 1 → Patients matching the given variant effect.
        - 0 → Other patients.

        Args:
            variant_effect_type (VariantEffect):
                Target variant effect class to evaluate.
            mane_tx_id (str or List[str]):
                MANE transcript ID(s) to filter variant effects.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]:
                - Binary matrix of variant effect classification.
                - Label mapping.
        """
        if not isinstance(variant_effect_type, VariantEffect):
            raise TypeError("variant_effect_type must be a VariantEffect enum")
        if isinstance(mane_tx_id, list) and not all(isinstance(tx, str) for tx in mane_tx_id):
            raise TypeError("mane_tx_id must be str or List[str]")
        label = str(variant_effect_type)

        cohort_creator = configure_caching_cohort_creator(self.hpo_hierarchy.hpo)
        cohort, _ = load_phenopackets(phenopackets=self.phenopackets, cohort_creator=cohort_creator)

        tx_list = [mane_tx_id] if isinstance(mane_tx_id, str) else mane_tx_id
        predicates = [variant_effect(variant_effect_type, tx_id=tx) for tx in tx_list]
        predicate = anyof(predicates)

        clf = monoallelic_classifier(
            a_predicate=predicate,
            b_predicate=~predicate,
            a_label=label,
            b_label="other"
        )

        variant_effects_matrix = pd.DataFrame(
            data=[1 if (cat := clf.test(p)) and cat.category.name == label else 0 for p in cohort],
            index=self.patient_index,
            columns=[label]
        )

        if variant_effects_matrix[label].sum() == 0:
            logger.warning(f"Warning: The column '{label}' in variant_effects_matrix is all zeros. Please check the corresponding mane_tx_id.")
    
        return variant_effects_matrix, {label: label}
    

    def _compute_target(
            self
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Computes the final target matrix and corresponding labels.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: 
            - Target matrix with disease status, sex, and optional variant effects
            - Mapping from column names to descriptive labels
        """
        disease_matrix= self._generate_disease_status_matrix()
        disease_matrix =disease_matrix.fillna(0)
        sex_matrix, sex_labels = self._generate_sex_matrix()

        variant_effects_matrix, variant_labels = None, {}
        if self.variant_effect_type and self.mane_tx_id:
            variant_effects_matrix, variant_labels = self._process_variant_effects(
                self.variant_effect_type, self.mane_tx_id
            )

        # Collect matrices to combine (note: disease_labels will be added later)
        matrices = [
            (disease_matrix, {}), # disease_labels to be added in the next step
            (sex_matrix, sex_labels),
        ]
        if variant_effects_matrix is not None:
            matrices.append((variant_effects_matrix, variant_labels))
        
        # Optional: Add external matrix if valid
        if self.external_target_matrix is not None:
            if not isinstance(self.external_target_matrix, pd.DataFrame):
                logger.warning("Warning: external_target_matrix is not a pandas DataFrame. It will be ignored.")
            else:
                external_matrix = self.external_target_matrix.reindex(index=self.patient_index)
                valid_values = {0, 1}
                invalid_mask = ~external_matrix.isin(valid_values) & ~external_matrix.isna()

                if invalid_mask.any().any():
                    invalid_entries = external_matrix[invalid_mask].stack()
                    logger.warning(
                        f"Warning: external_target_matrix contains invalid values (only 0, 1, and NaN are allowed).\n"
                        f"The following invalid entries were found and the matrix will be skipped:\n{invalid_entries}"
                    )
                else:
                    matrices.append((external_matrix, {col: col for col in external_matrix.columns}))
        # Combine and return
        combined_matrix = pd.concat([m.reindex(self.patient_index) for m, _ in matrices], axis=1)
        combined_labels = {k: v for _, label_dict in matrices for k, v in label_dict.items()}

        return combined_matrix, combined_labels




