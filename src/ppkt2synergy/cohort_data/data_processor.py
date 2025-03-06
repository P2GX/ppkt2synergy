from typing import List, Dict, Set, Union, IO
from .hpo_utils import load_hpo
from .matrix_generator import CohortMatrixGenerator
import pandas as pd

class CohortDataProcessor:
    def __init__(
        self,  
    ):
        """
        Processes cohort data matrices through hierarchical feature selection.
    
        Provides methods to filter and structure HPO term matrices based on their 
        ontological relationships.
        
        Example:
            from ppkt2synergy import CohortDataLoader, CohortMatrixGenerator
            # Load data and generate matrices
            dataloader = CohortDataLoader.from_ppkt_store('FBN1')
            matrix_gen = CohortMatrixGenerator(dataloader.phenopackets)
            # Preprocess matrices
            hpo_mat, disease_mat = CohortDataProcessor.preprocess_matrices(
                 matrix_gen, 
                 threshold=0.8, 
                 mode='root'
           )

        """
    
    @staticmethod
    def matrix_preprocessing(
        data_generator:CohortMatrixGenerator, 
        threshold=0.5, 
        mode='leaf',
        hpo_file: Union[IO, str] = None, 
        use_label: bool = True
        ) -> pd.DataFrame:
        """
        Filters and structures the HPO term matrix based on term hierarchy.
        
        Args:
            data_generator: Instance of CohortMatrixGenerator containing HPO and disease matrices.
            threshold: Maximum allowed NaN ratio for a column to be retained.
            mode: 'leaf' to extract leaf terms, 'root' to extract root terms.
            hpo_file: Path or URL to the HPO ontology file.
            use_label: Whether to replace term IDs with labels.
        
        Returns:
            A tuple containing the processed HPO and disease matrices.
        """ 

         # Validate inputs
        if not 0 <= threshold <= 1:
            raise ValueError(f"NaN threshold {threshold} must be 0 - 1")
        if mode not in {'leaf', 'root'}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'leaf'/'root'")
        
        hpo_matrix = data_generator.hpo_term_observation_matrix
        hpo_matrix_filtered = hpo_matrix.dropna(axis=1, thresh=int(threshold * len(hpo_matrix)))
        classifier = HPOTermClassifier(hpo_file)   
        subtrees = classifier.classify_terms(set(hpo_matrix_filtered.columns))

        select_terms = []
        for root, data in subtrees.items():
            leaves = data["leaves"]
            # If mode is 'root', only extract the root term(s) (root itself)
            if mode == 'root':
                select_terms.extend([root])  
            # If mode is 'leaf', extract only the leaf terms or a single term if there's only one
            elif mode == 'leaf':
                select_terms.extend(leaves)

        selected_columns = list(set(select_terms))
        if not selected_columns:
            raise ValueError("No valid columns selected after filtering. Please check your threshold and mode settings.")

        final_matrix = hpo_matrix_filtered[selected_columns].fillna(0)
        disease_matrix = data_generator.disease_matrix.fillna(0)

        if use_label:
            final_matrix = final_matrix.rename(columns= data_generator.hpo_labels)
            disease_matrix = disease_matrix.rename(columns= data_generator.disease_labels)
        

        return final_matrix, disease_matrix 


class HPOTermClassifier:
    """
    Classifies HPO terms into hierarchical subtrees, identifying roots and leaves within each subtree.
    """

    def __init__(self, hpo_file: Union[IO, str] = None):
        """
        Initializes the classifier by loading the HPO ontology.

        Args:
            hpo_file: The URL or path to load the HPO. If None, the latest HPO will be loaded.
        """
        try:
            self.hpo = load_hpo(hpo_file)
        except Exception as e:
            raise ValueError(f"Error loading HPO file: {e}")


    def classify_terms(self, terms: Set[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Groups HPO terms into subtrees and extracts root and leaf terms.

        Args:
            terms: A set of HPO term IDs.
        
        Returns:
            A dictionary mapping subtree root IDs to their terms and leaves.
        """

        roots = self._find_roots(terms)
        subtrees = {root: {"terms": list(self._get_subtree_terms(root, terms)),
                           "leaves": self._extract_leaves(root, terms)}
                    for root in roots}
        return subtrees


    def _find_roots(self, terms: Set[str]) -> List[str]:
        """
        Identifies root terms (terms without ancestors in the given set).
        """
        roots = [term for term in terms if not any(a.value in terms for a in self.hpo.graph.get_ancestors(term))]
        return roots


    def _get_subtree_terms(self, root: str, terms: Set[str]) -> Set[str]:
        """
        Retrieves all terms in the subtree rooted at a given term.
        """
        subtree_terms = set()
        stack = [root]
        while stack:
            current_term = stack.pop()
            if current_term not in subtree_terms:
                subtree_terms.add(current_term)
                stack.extend(t.value for t in self.hpo.graph.get_descendants(current_term) if t.value in terms)
        return subtree_terms


    def _extract_leaves(self, root: str, terms: Set[str]) -> List[str]:
        """
        Extracts leaf terms (terms without descendants in the given set).
        """
        subtree_terms = self._get_subtree_terms(root, terms)
        leaves = [term for term in subtree_terms if not any(d.value in terms for d in self.hpo.graph.get_descendants(term))]
        return leaves
    



    


        
        



        
    

    




    

