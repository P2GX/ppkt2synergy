from typing import List, Dict, Set, Union, IO, Optional, Tuple
from ..io import load_hpo,CohortDataLoader
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HPOHierarchyUtils:
    """
    Utility class for handling hierarchical relationships between HPO terms.

    Main functionalities:
    1. Propagate observed (1) and excluded (0) HPO term annotations in a sample × HPO matrix,
    respecting the HPO ontology structure.
    2. Classify a given set of HPO terms into subtrees rooted at top-level terms.
    3. Identify hierarchical conflicts in observed/excluded data.
    4. Provide utility functions for hierarchical masking and term labeling.
        print(result)
        # Output:
        # {
        #     'HP:0004322': {'terms': ['HP:0004322', 'HP:0012758'], 'leaves': ['HP:0012758']},
        #     'HP:0001250': {'terms': ['HP:0001250'], 'leaves': ['HP:0001250']}
        # }
    """

    def __init__(
            self,
            hpo_file: Optional[Union[str, IO]] = None
        ):
        """
        Initialize the HPOHierarchyUtils with an HPO ontology.

        Args:
            hpo_file: One of the following:
                - Path to an HPO OBO file (.obo or .json)
                - Open file-like object
                - None (loads the latest HPO version by default)

        Raises:
            ValueError: If loading the HPO ontology fails.
        """
        try:
            self.hpo = load_hpo(hpo_file)
        except Exception as e:
            raise ValueError(f"Failed to load HPO ontology: {e}")
            
        # Cache ancestors and descendants of terms to avoid repeated queries
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._descendant_cache: Dict[str, Set[str]] = {}
        self._current_terms: Set[str] = set()

    def _validate_term(
            self, 
            term: str
        ) -> bool:
        """Check if term exists in ontology."""
        return term in self.hpo
    

    def _cache_term_relations(
            self, 
            terms: Set[str]
        ) -> None:
        """
        Incrementally prepare term caches:
        - Add only new, valid terms.
        - Cache their ancestors and descendants.
        - Update self._current_terms without overwriting existing ones.
        """
        self._current_terms = {t for t in terms if self._validate_term(t)} 

        for term in self._current_terms:
            if term not in self._ancestor_cache:
                try:
                    self._ancestor_cache[term] = {a.value for a in self.hpo.graph.get_ancestors(term)}
                except Exception as e:
                    logger.warning(f"Warning: failed to get ancestors for term {term}: {e}")
                    self._ancestor_cache[term] = set()
            if term not in self._descendant_cache:
                try:
                    self._descendant_cache[term] = {d.value for d in self.hpo.graph.get_descendants(term)}
                except Exception as e:
                    logger.warning(f"Warning: failed to get descendants for term {term}: {e}")
                    self._descendant_cache[term] = set()
    

    def propagate_hpo_hierarchy(
            self, 
            matrix: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Applies hierarchical propagation to an HPO term matrix.

        Propagation logic:
        - A value of 1 (observed) propagates to all ancestors of the corresponding HPO term.
        - A value of 0 (excluded) propagates to all descendants.

        If a term is not found in the ontology, it is removed from the matrix.

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
            matrix (pd.DataFrame): 
                Input matrix with raw HPO observations/exclusions..

        Returns:
            pd.DataFrame: 
                Matrix with propagated values.
        """
        terms = set(matrix.columns)
        self._cache_term_relations(terms)

        invalid_terms = terms - self._current_terms
        if invalid_terms:
            matrix = matrix.copy().drop(columns=invalid_terms)

        for term in self._current_terms:
            # ---- Propagate observed (1) values upwards ----
            ancestors = self._ancestor_cache.get(term, set())
            valid_ancestors = ancestors & self._current_terms
            if valid_ancestors:
                mask = matrix[term] == 1
                for anc in valid_ancestors:
                    conflict_mask = mask & (matrix[anc] == 0)
                    if conflict_mask.any():
                        conflict_indices = matrix.index[conflict_mask].tolist()
                        print(f"[Conflict] Term {anc} is an ancestor of {term}, but in these samples {term}=1 and {anc}=0: {conflict_indices}")
                    # Only assign where target is NaN
                    update_mask = mask & (matrix[anc].isna())
                    matrix.loc[update_mask, anc] = 1

            # ---- Propagate excluded (0) values downwards ----
            descendants = self._descendant_cache.get(term, set())
            valid_descendants = descendants & self._current_terms
            if valid_descendants:
                mask = matrix[term] == 0
                for desc in valid_descendants:
                    conflict_mask = mask & (matrix[desc] == 1)
                    if conflict_mask.any():
                        conflict_indices = matrix.index[conflict_mask].tolist()
                        print(f"[Conflict] Term {desc} is a descendant of {term}, but in these samples {term}=0 and {desc}=1: {conflict_indices}")
                    # Only assign where target is NaN
                    update_mask = mask & (matrix[desc].isna())
                    matrix.loc[update_mask, desc] = 0


        return matrix 
 

    def classify_terms(
            self, 
            terms: Set[str]
        ) -> Dict[str, Dict[str, List[str]]]:
        """
        Classify given HPO terms into subtrees defined by root terms.

        Root terms: terms with no ancestors within the set.
        Leaf terms: terms with no descendants within the set.

        Args:
            terms(Set[str]): 
                A set of HPO term IDs.

        Returns:
            Dict[str, Dict[str, List[str]]]: 
                - "terms": All terms in the subtree.
                - "leaves": Leaf terms in the subtree.
        """
        self._cache_term_relations(terms)
        roots = self._find_roots(terms)
        results = {}
        for root in roots:
            subtree = self._get_subtree_terms(root, terms)
            results[root] = {
                "terms": sorted(list(subtree)),
                "leaves": sorted(self._extract_leaves(subtree, terms))
            }
        return results

    def _find_roots(
            self, 
            terms: Set[str]
        ) -> List[str]:
        """ Finds root terms (terms with no ancestors in the given set). """
        return [t for t in terms if not any(a in terms for a in self._ancestor_cache.get(t, set()))]

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
                stack.extend(d for d in self._descendant_cache.get(term, set()) if d in terms)
        return subtree

    def _extract_leaves(
            self, 
            subtree: Set[str], 
            terms: Set[str]
        ) -> List[str]:
        """ Finds leaf terms (terms with no descendants in the given set). """
        return [t for t in subtree if not any(d in terms for d in self._descendant_cache.get(t, set()))]
    
    

    def build_relationship_mask(
            self, 
            terms: List[str]
        ) -> pd.DataFrame:
        """
        Build a term × term mask matrix where each cell is set to NaN if the two terms
        have a hierarchical relationship (i.e., one is an ancestor or descendant of the other),
        and 0 otherwise.

        Args:
            terms (List): A list of HPO term IDs to include in the mask.

        Returns:
            pd.DataFrame: A square DataFrame (terms × terms) where:
                        - cell (i, j) is NaN if term_i and term_j are hierarchically related;
                        - otherwise, the value is 0.
        """
        terms = list(terms)
        # Initialize a matrix filled with 0s; it will be updated to 1 or NaN later.
        mask = pd.DataFrame(0, index=terms, columns=terms, dtype=float)

        # Ensure ancestor and descendant relationships are cached for these terms
        self._cache_term_relations(terms)

        nan_positions = set()
        for term in terms:
            related = (self._ancestor_cache[term] | self._descendant_cache[term]) & set(terms)
            nan_positions.update({(term, rel) for rel in related})
            nan_positions.update({(rel, term) for rel in related})
            nan_positions.add((term, term))
        
        for r, c in nan_positions:
            mask.loc[r, c] = np.nan

        return mask
    

    def create_hpo_and_disease_labels(
            self,
        ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build up-to-date HPO and disease label mappings from phenopackets using the latest HPO ontology.

        Returns:
            A tuple of two dictionaries:
                - hpo_labels: A dictionary mapping HPO IDs to term labels.
                - disease_labels: A dictionary mapping disease IDs to disease names (from phenopackets).
        """
        all_phenopackets = CohortDataLoader.from_ppkt_store()

        hpo_labels = {}
        disease_labels = {}

        for ppkt in all_phenopackets:
            for feature in ppkt.phenotypic_features:
                if feature.type and feature.type.id:
                    hpo_id = feature.type.id
                    term_name = self.hpo.get_term_name(hpo_id)
                    if term_name:
                        hpo_labels[hpo_id] = term_name
                    else:
                        logger.warning(f"HPO term not found in ontology: {hpo_id} — using ID as label")
                        hpo_labels[hpo_id] = hpo_id  # fallback to ID

            for disease in ppkt.diseases:
                if disease.term and disease.term.id:
                    disease_id = disease.term.id
                    label = disease.term.label or disease_id
                    if disease.term.label is None:
                        logger.warning(f"Disease label missing for: {disease_id} — using ID as label")
                    disease_labels[disease_id] = label

        return hpo_labels, disease_labels



   


    


        
        



        
    

    




    

