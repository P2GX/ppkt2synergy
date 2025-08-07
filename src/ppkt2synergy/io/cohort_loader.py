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
        cohort_name: Optional[Union[str, List[str]]]=None,
        ppkt_store_version: Optional[str] = None,
    ) -> List[ppkt.Phenopacket]:
        """
        Load Phenopacket objects for one or more cohorts from the configured Phenopacket Store.

        Args:
            cohort_name (Union[str, List[str], None]):
                A single cohort name, a list of cohort names, or None.
                If None, phenopackets from all cohorts in the store will be loaded.  

            ppkt_store_version (Optional[str]):  (default: None)
                A string specifying the release tag of the Phenopacket Store (e.g., `'0.1.23'`).  
                If `None`, the latest available release will be used.
        Returns:
            List[phenopackets.Phenopacket]: 
                A list of Phenopacket objects corresponding to the specified cohort(s).
        """
        registry = configure_phenopacket_registry()
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            if cohort_name is None:
                # Load all phenopackets from the store
                cohort_names = [cohort.name for cohort in ps.cohorts()]  
            elif isinstance(cohort_name, str):
                cohort_names = [cohort_name]
            else:
                cohort_names = cohort_name
            phenopackets = []
            for name in cohort_names:
                phenopackets.extend(list(ps.iter_cohort_phenopackets(name))) 
        return phenopackets
    
    @staticmethod
    def partition_phenopackets_by_cohorts(
        cohort_name: Union[str, List[str]],
        ppkt_store_version: Optional[str] = None,
    ) -> tuple[List[ppkt.Phenopacket], List[ppkt.Phenopacket]]:
        """
        Partition Phenopackets from the Phenopacket Store into two groups:
        - Those belonging to the specified cohort(s)
        - The rest of the Phenopackets

        Args:
            cohort_name (Union[str, List[str]]):  
                A single cohort name or a list of cohort names to extract.

            ppkt_store_version (Optional[str]):  (default: None)
                Release tag of the Phenopacket Store (e.g., '0.1.23'). If None, use latest.

        Returns:
            Tuple[List[Phenopacket], List[Phenopacket]]: 
                A tuple containing:
                - The Phenopackets from the specified cohort(s)
                - The remaining Phenopackets
        """
        registry = configure_phenopacket_registry()
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            all_ppkts = []
            target_ppkts = []

            if isinstance(cohort_name, str):
                target_names = [cohort_name]
            else:
                target_names = cohort_name

            # Collect all phenopackets
            for cohort in ps.cohorts():
                name = cohort.name
                ppkts = list(ps.iter_cohort_phenopackets(name))
                if name in target_names:
                    target_ppkts.extend(ppkts)
                else:
                    all_ppkts.extend(ppkts)

        return target_ppkts, all_ppkts
