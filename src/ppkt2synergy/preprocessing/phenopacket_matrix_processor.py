from typing import List, Union, IO, Optional,Tuple
from .phenopacket_matrix_generator import PhenopacketMatrixGenerator
import pandas as pd
import phenopackets as ppkt
from gpsea.model import VariantEffect
from ._utils import HPOHierarchyUtils

class PhenopacketMatrixProcessor:
    """
    Processes HPO term observation matrices for downstream analyses such as clustering, classification, or synergy computation.

    Key Features:
    - Filters HPO terms based on missing value thresholds and hierarchical structure (root or leaf selection).
    - Supports mapping from HPO IDs to human-readable labels.
    - Prepares both HPO term matrix and disease target matrix from phenopackets.
    - Enables one-hot or binary target extraction for downstream tasks.

    Example:
        from ppkt2synergy import CohortDataLoader, HPOMatrixProcessor

        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> hpo_matrix, target_matrix = PhenopacketMatrixProcessor.prepare_hpo_data(
                phenopackets,
                threshold=0.5,
                mode='leaf',
                use_label=True,
            target_name=None)
    """

    def __init__(self):
        pass

    @staticmethod
    def prepare_hpo_data(
            phenopackets: List[ppkt.Phenopacket], 
            hpo_file: Union[IO, str] = None,
            variant_effect_type: Optional[VariantEffect] = None,
            mane_tx_id: Optional[Union[str, List[str]]] = None,
            external_target_matrix: Optional[pd.DataFrame] = None, 
            threshold: float = 0.5, 
            mode: str = None,
            use_label: bool = True,
            nan_strategy=None,
        ) -> Tuple[Tuple[pd.DataFrame,Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Prepare and filter an HPO term matrix and a target matrix from a list of phenopackets.

        This function generates a binary matrix of observed HPO terms for each patient,
        applies optional filtering based on ontology hierarchy (e.g., leaf or root terms),
        handles missing values, and returns the final observation and target matrices.

        Args:
            phenopackets (List[Phenopacket]): 
                List of phenopackets to generate observation matrices.
            hpo_file (Union[IO, str], optional): (default: None)
                Path or URL to the HPO ontology file. 
            variant_effect_type (Optional[str]): (default: None)
                The specific variant effect class to evaluate.
                This should be a member of the `VariantEffect` Enum from `gpsea.model`,
                for example, `VariantEffect.MISSENSE_VARIANT` or `VariantEffect.NONSENSE_VARIANT`. 
            mane_tx_id (Optional[str or List[str]]): (default: None)
                MANE transcript ID(s) used to filter variant effects. 
                Different cohorts may use different transcripts, so providing one or more IDs ensures consistent variant effect classification across datasets. 
            external_target_matrix (Optional[pd.DataFrame]): (default: None)
                User-provided binary matrix (indexed by patient ID). 
                Columns with only valid values (0, 1, or NaN) are merged into the final target matrix; others are skipped with a warning. 
            threshold (float): (default: 0.5)
                Maximum allowed proportion of NaN values. Columns exceeding this threshold are dropped. 
            mode (str, default 'leaf'): (default: 'None')
                Determines which HPO terms to retain based on ontology hierarchy.
                - 'leaf': keeps the most specific terms (i.e., those without descendants in the selected set).
                - 'root': keeps the most general terms (i.e., those without ancestors in the selected set). 
                -  None: keeps all terms.
            use_label (bool): (default: True)
                Whether to replace term IDs with their labels (if available). 
            nan_strategy (str, optional): (default: None)
                Strategy for handling missing values.
                - "impute_zero": fill NaNs with 0
                - None: keep NaNs for downstream analysis
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
            - `final_matrix`: filtered HPO observation matrix (patients × HPO terms)
            - `target_matrix`: binary target matrix (patients × labels or variant effects)
            - `relationship_mask` (optional): if `mode` is None, returns a term × term DataFrame where:
                - entries are `NaN` if two terms are hierarchically related (ancestor/descendant)
                - entries are `1` if the terms are independent (no hierarchical relationship)
              Otherwise, returns `None`.

        Raises:
            ValueError: 
                - If threshold is not between 0 and 1.
                - If `mode` is not one of {"leaf", "root", None}.
                - If `nan_strategy` is invalid.
                - If no valid terms remain after filtering.
        """
        
        if not 0 <= threshold <= 1:
            raise ValueError(f"NaN threshold {threshold} must be between 0 and 1")
        if mode not in {'leaf', 'root', None}:
            raise ValueError(f"Invalid mode: '{mode}'. Choose 'leaf' or 'root' or None.")
        
        classifier = HPOHierarchyUtils(hpo_file)

        data_generator = PhenopacketMatrixGenerator(
            phenopackets=phenopackets,
            hpo_file=hpo_file,
            variant_effect_type=variant_effect_type,
            mane_tx_id=mane_tx_id,
            use_labels=False,
            external_target_matrix=external_target_matrix,
            hpo_hierarchy = classifier) 
        
        hpo_matrix = data_generator.hpo_term_observation_matrix
        target_matrix = data_generator.target_matrix

        hpo_matrix_filtered = hpo_matrix.dropna(axis=1, thresh=int(threshold * len(hpo_matrix)))

        if mode is None:
            selected_columns = list(hpo_matrix_filtered.columns.values)
            relationship_mask  = classifier.build_relationship_mask(hpo_matrix_filtered.columns)
        else:
            selected_columns = PhenopacketMatrixProcessor._select_terms_by_hierarchy(hpo_matrix_filtered, classifier, mode)

        if not selected_columns :
            raise ValueError("No valid terms found. Adjust threshold or mode.")

        if nan_strategy == "impute_zero":
            final_matrix = hpo_matrix_filtered[selected_columns].fillna(0)
            target_matrix = target_matrix.fillna(0)
        elif nan_strategy is None:
            final_matrix = hpo_matrix_filtered[selected_columns]
        else :
            raise ValueError(f"Invalid nan_strategy: {nan_strategy}. Use 'impute_zero' or None.")

        # Replace term IDs with labels 
        if use_label:
            final_matrix = PhenopacketMatrixProcessor._apply_hpo_labels(final_matrix, data_generator)
            target_matrix = PhenopacketMatrixProcessor._apply_hpo_labels(target_matrix, data_generator)
            relationship_mask = PhenopacketMatrixProcessor._apply_hpo_labels(relationship_mask, data_generator) if mode is None else None

        return ((final_matrix, relationship_mask  if mode is None else None ), target_matrix) 


    @staticmethod
    def _select_terms_by_hierarchy(
            hpo_matrix: pd.DataFrame,
            classifier: HPOHierarchyUtils, 
            mode: str = 'leaf'
        ) -> List[str]:
        """
        Selects valid HPO terms based on hierarchy (root/leaf).

        Args:
            hpo_matrix (pd.DataFrame): 
                HPO observation matrix (patients * HPO terms).
            classifier (HPOHierarchyClassifier): 
                Classifies terms into subtrees based on ontology structure.
            mode (str): (default: 'leaf')
                -"root" to select general terms, 
                -"leaf" to select specific terms.

        Returns:
            List[str]: Names of selected HPO term columns.
        """
        subtrees = classifier.classify_terms(set(hpo_matrix.columns))
        select_terms = []
        for root, data in subtrees.items():
            if mode == 'root':
                select_terms.append(root)
            elif mode == 'leaf':
                select_terms.extend(data["leaves"])
        return list(set(select_terms))


    @staticmethod
    def _apply_hpo_labels(
            matrix: pd.DataFrame, 
            data_generator: PhenopacketMatrixGenerator
        ) -> pd.DataFrame:
        """
        Replaces HPO term IDs with corresponding labels (if available).

        Args:
            matrix (pd.DataFrame): 
                The data matrix.
            data_generator (PhenopacketMatrixGenerator): 
                Provides HPO labels.

        Returns:
            pd.DataFrame: Matrix with IDs replaced by labels.
        """
        label_mapping = {**data_generator.hpo_labels, **data_generator.target_labels}
        # Apply the combined mapping
        matrix = matrix.rename(columns=label_mapping)
        return matrix



