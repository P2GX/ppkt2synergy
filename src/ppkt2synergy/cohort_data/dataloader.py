import typing
import phenopackets as ppkt
from ppktstore.registry import configure_phenopacket_registry


class CohortDataLoader:
    """
    A class to load cohort data (Phenopacket objects) from a Phenopacket Store.
    """

    def __init__(self):
        pass

    @staticmethod
    def from_ppkt_store(
        cohort_name: str, 
        ppkt_store_version: typing.Optional[str] = None
        ) -> typing.List[ppkt.Phenopacket]:
        """
        Retrieve Phenopacket objects for a specific cohort from a Phenopacket Store.

        Args:
            cohort_name (str): The name of a cohort in Phenopacket Store
            ppkt_store_version (str):  a `str` with Phenopacket Store release tag (e.g. `0.1.23`) or `None`
            if the *latest* release should be loaded.

        Returns:
            a list of GA4GH Phenopacket objects for the cohort
        """
        registry = configure_phenopacket_registry()
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            phenopackets = list(ps.iter_cohort_phenopackets(cohort_name))

        # Check if the result is empty and raise an exception if necessary
        if not phenopackets:
            raise ValueError(f"No phenopackets found for cohort '{cohort_name}' in version '{ppkt_store_version}'.")
        
        return phenopackets