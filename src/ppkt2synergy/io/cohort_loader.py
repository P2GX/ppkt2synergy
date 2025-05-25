from typing import Union,List, Optional
import phenopackets as ppkt
from ppktstore.registry import configure_phenopacket_registry


class CohortDataLoader:
    """
    A utility class for loading GA4GH Phenopacket objects from a Phenopacket Store.

    This class provides methods to access phenotypic data grouped by cohort names, 
    useful for downstream analysis in biomedical and genomic research.
    """

    def __init__(self):
        pass

    @staticmethod
    def from_ppkt_store(
        cohort_name: Union[str, List[str]],
        ppkt_store_version: Optional[str] = None,
    ) -> List[ppkt.Phenopacket]:
        """
        Load Phenopacket objects for one or more cohorts from the configured Phenopacket Store.

        Args:
            cohort_name (Union[str, List[str]]):  
                A single cohort name or a list of cohort names to be loaded.  

            ppkt_store_version (Optional[str]):  (default: None)
                A string specifying the release tag of the Phenopacket Store (e.g., `'0.1.23'`).  
                If `None`, the latest available release will be used.
        Returns:
            List[phenopackets.Phenopacket]: 
                A list of Phenopacket objects corresponding to the specified cohort(s).
        """
        registry = configure_phenopacket_registry()
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            if isinstance(cohort_name, str):
                cohort_names = [cohort_name]
            else:
                cohort_names = cohort_name

            phenopackets = []
            for name in cohort_names:
                phenopackets.extend(list(ps.iter_cohort_phenopackets(name))) 
        return phenopackets