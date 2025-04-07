from typing import List, Dict, Set, Union, IO
from .hpo_utils import load_hpo
from .matrix_generator import CohortMatrixGenerator
import pandas as pd

class HPOMatrixProcessor:
    """
    Filters HPO term observation matrices based on hierarchical relationships and data quality.

    Key Features:
    - Removes terms with excessive missing values.
    - Selects terms by hierarchical level (root or leaf nodes).
    - Optionally replaces term IDs with human-readable labels.

    Example:
        >>> matrix_gen = CohortMatrixGenerator(phenopackets)
        >>> hpo_mat, disease_mat = HPOTermFilter.filter_by_hierarchy(
        ...     matrix_gen, max_na_ratio=0.3, term_level='leaf')
    """

    @staticmethod
    def filter_hpo_matrix(
        data_generator: CohortMatrixGenerator, 
        threshold: float = 0.5, 
        mode: str = 'leaf',
        hpo_file: Union[IO, str] = None, 
        use_label: bool = True
    ) -> pd.DataFrame:
        """
        Filters the HPO term matrix by selecting features based on term hierarchy (root/leaf).

        Args:
            data_generator (CohortMatrixGenerator): 
                An instance generating the HPO term observation matrix and disease matrix.
            threshold (float, default 0.5): 
                Maximum allowed proportion of NaN values. Columns exceeding this threshold are dropped.
                (e.g., threshold=0.5 means columns with more than 50% missing values will be removed).
            mode (str, default 'leaf'): 
                Select "root" to retain root terms or "leaf" to retain leaf terms.
            hpo_file (Union[IO, str], optional): 
                Path or URL to the HPO ontology file.
            use_label (bool, default True): 
                Whether to replace term IDs with their labels (if available).

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

        hpo_matrix = data_generator.hpo_term_observation_matrix
        target_matrix = data_generator.target_matrix

        # Align sample indices first
        common_index = hpo_matrix.index.intersection(target_matrix.index)
        hpo_matrix = hpo_matrix.loc[common_index]
        target_matrix = target_matrix.loc[common_index]

        hpo_matrix_filtered = hpo_matrix.dropna(axis=1, thresh=int(threshold * len(hpo_matrix)))


        classifier = HPOHierarchyClassifier(hpo_file)
        selected_columns = HPOMatrixProcessor._select_terms_by_hierarchy(hpo_matrix_filtered, classifier, mode)

        if not selected_columns:
            raise ValueError("No valid terms found. Adjust threshold or mode.")

        final_matrix = hpo_matrix_filtered[selected_columns].fillna(0)

        # Replace term IDs with labels (if enabled)
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
        return matrix.rename(columns=data_generator.hpo_labels)



class HPOHierarchyClassifier:
    """
    Identifies root and leaf terms in HPO term subtrees.
    
    Example:
        classifier = HPOHierarchyClassifier("path/to/hpo.obo")
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
        return {
            root: {
                "terms": list(self._get_subtree_terms(root, terms)),
                "leaves": self._extract_leaves(root, terms)
            }
            for root in roots
        }

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
            root: str, 
            terms: Set[str]
            ) -> List[str]:
        """ Finds leaf terms (terms with no descendants in the given set). """
        subtree = self._get_subtree_terms(root, terms)
        return [t for t in subtree if not any(d.value in terms for d in self.hpo.graph.get_descendants(t))]

    



    


        
        



        
    

    




    

