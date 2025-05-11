from typing import List, Dict, Set, Union, IO, Optional,Tuple
from ._utils import load_hpo
from .matrix_generator import PhenopacketMatrixGenerator
import pandas as pd
import phenopackets as ppkt
from gpsea.model import VariantEffect

class HPOMatrixProcessor:
    """
    Filters HPO term observation matrices based on hierarchical relationships and data quality.

    Key Features:
    - Removes terms with excessive missing values.
    - Selects terms by hierarchical level (root or leaf nodes).
    - Optionally replaces term IDs with human-readable labels.

    Example:
        from ppkt2synergy import CohortDataLoader, HPOMatrixProcessor
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> hpo_matrix, target_matrix = HPOMatrixProcessor.prepare_hpo_data(
            phenopackets,  threshold=0.5, mode='leaf', use_label=True)
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
        mode: str = 'leaf',
        use_label: bool = True,
        nan_strategy=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters the HPO term matrix by selecting features based on term hierarchy (root/leaf).

        Args:
            phenopackets (List[Phenopacket]): 
                List of phenopackets to generate observation matrices.
            hpo_file (Union[IO, str], optional): 
                Path or URL to the HPO ontology file.
            variant_effect_type (Optional[VariantEffect]): 
                Type of variant effect to filter by (optional).
            mane_tx_id (Optional[str or List[str]]): 
                Specific transcript ID(s) to filter variants.
            external_target_matrix (Optional[pd.DataFrame]): 
                Optional external matrix representing targets.
            threshold (float, default 0.5): 
                Maximum allowed proportion of NaN values. Columns exceeding this threshold are dropped.
            mode (str, default 'leaf'): 
                Select "root" to retain root terms or "leaf" to retain leaf terms.
            use_label (bool, default True): 
                Whether to replace term IDs with their labels (if available).
            nan_strategy (str, optional):
                Strategy for handling missing values.
                - "fill": fill NaNs with 0
                - "drop": drop rows with any NaNs
                - None: do not handle missing values

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - Processed HPO term matrix (`hpo_matrix`)
                - Processed disease matrix (`disease_matrix`)

        Raises:
            ValueError: 
                - If threshold is not between 0 and 1.
                - If mode is not 'leaf' or 'root'.
        """
        
        if not 0 <= threshold <= 1:
            raise ValueError(f"NaN threshold {threshold} must be between 0 and 1")
        if mode not in {'leaf', 'root'}:
            raise ValueError(f"Invalid mode: '{mode}'. Choose 'leaf' or 'root'.")
        
        data_generator = PhenopacketMatrixGenerator(phenopackets=phenopackets,hpo_file=hpo_file,variant_effect_type=variant_effect_type,mane_tx_id=mane_tx_id,external_target_matrix=external_target_matrix,use_labels=False) 
        
        hpo_matrix = data_generator.hpo_term_observation_matrix
        target_matrix = data_generator.target_matrix

        hpo_matrix_filtered = hpo_matrix.dropna(axis=1, thresh=int(threshold * len(hpo_matrix)))

        classifier = HPOHierarchyClassifier(hpo_file)
        selected_columns = HPOMatrixProcessor._select_terms_by_hierarchy(hpo_matrix_filtered, classifier, mode)

        if not selected_columns:
            raise ValueError("No valid terms found. Adjust threshold or mode.")

        if nan_strategy == "fill":
            final_matrix = hpo_matrix_filtered[selected_columns].fillna(0)
        elif nan_strategy == "drop":
            final_matrix = hpo_matrix_filtered[selected_columns].dropna(axis=0)
            target_matrix = target_matrix.loc[final_matrix.index]
        elif nan_strategy is None:
            final_matrix = hpo_matrix_filtered[selected_columns]
        else :
            raise ValueError(f"Invalid nan_strategy: {nan_strategy}. Use 'fill', 'drop', or None.")
        target_matrix = target_matrix.fillna(0)

        # Replace term IDs with labels 
        if use_label:
            final_matrix = HPOMatrixProcessor._apply_hpo_labels(final_matrix, data_generator)
            target_matrix = HPOMatrixProcessor._apply_hpo_labels(target_matrix, data_generator)

        return final_matrix, target_matrix 

    @staticmethod
    def _select_terms_by_hierarchy(
        hpo_matrix, 
        classifier, mode
        ) -> List[str]:
        """
        Selects valid HPO terms based on hierarchy (root/leaf).

        Args:
            hpo_matrix (pd.DataFrame): HPO observation matrix.
            classifier (HPOHierarchyClassifier): HPO hierarchy classifier.
            mode (str): "root" or "leaf".

        Returns:
            List[str]: Filtered term columns.
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
        matrix, 
        data_generator
        ) -> pd.DataFrame:
        """
        Replaces HPO term IDs with corresponding labels (if available).

        Args:
            matrix (pd.DataFrame): The data matrix.
            data_generator (CohortMatrixGenerator): Provides HPO labels.

        Returns:
            pd.DataFrame: Matrix with IDs replaced by labels.
        """
        label_mapping = {**data_generator.hpo_labels, **data_generator.target_labels}
        # Apply the combined mapping
        matrix = matrix.rename(columns=label_mapping)
        return matrix



class HPOHierarchyClassifier:
    """
    Identifies root and leaf terms in HPO term subtrees.
    - Each root is the top-most term in a connected subtree (no ancestors in given set).
    - If a root has no children, it is considered a leaf itself.
    
    Example:
        classifier = HPOHierarchyClassifier()
        result = classifier._get_subtree_terms({"HP:0004322", "HP:0001250", "HP:0012758"})
        print(result)
        # Output:
        # {
        #     'HP:0004322': {'terms': ['HP:0004322', 'HP:0012758'], 'leaves': ['HP:0012758']},
        #     'HP:0001250': {'terms': ['HP:0001250'], 'leaves': ['HP:0001250']}
        # }
    """

    def __init__(
            self, 
            hpo_file: Union[IO, str] = None
            ):
        """
        Loads the HPO ontology.

        Args:
            hpo_file: Path or URL to the HPO file (default: latest version).
        """
        try:
            self.hpo = load_hpo(hpo_file)
        except Exception as e:
            raise ValueError(f"Failed to load HPO file: {e}")

    def classify_terms(
            self, 
            terms: Set[str]
            ) -> Dict[str, Dict[str, List[str]]]:
        """
        Groups HPO terms into subtrees with their root and leaf terms.

        Args:
            terms: A set of HPO term IDs.

        Returns:
            Dict[str, Dict[str, List[str]]]: 
                - "terms": All terms in the subtree.
                - "leaves": Leaf terms in the subtree.
        """
        roots = self._find_roots(terms)
        results = {}
        for root in roots:
            subtree = self._get_subtree_terms(root, terms)
            if len(subtree) == 1:
                results[root] = {
                    "terms": list(subtree),
                    "leaves": list(subtree)
                }
            else:
                results[root] = {
                    "terms": list(subtree),
                    "leaves": self._extract_leaves(subtree, terms)
                }
        return results

    def _find_roots(
            self, 
            terms: Set[str]
            ) -> List[str]:
        """ Finds root terms (terms with no ancestors in the given set). """
        return [t for t in terms if not any(a.value in terms for a in self.hpo.graph.get_ancestors(t))]

    def _get_subtree_terms(
            self, 
            root: str, 
            terms: Set[str]
            ) -> Set[str]:
        """ Collects all terms under a given root. """
        subtree = set()
        stack = [root]
        while stack:
            term = stack.pop()
            if term not in subtree:
                subtree.add(term)
                stack.extend(t.value for t in self.hpo.graph.get_descendants(term) if t.value in terms)
        return subtree

    def _extract_leaves(
            self, 
            subtree: Set[str], 
            terms: Set[str]
            ) -> List[str]:
        """ Finds leaf terms (terms with no descendants in the given set). """
        return [t for t in subtree if not any(d.value in terms for d in self.hpo.graph.get_descendants(t))]

    



    


        
        



        
    

    




    

