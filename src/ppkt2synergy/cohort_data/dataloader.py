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
        cohort_name: typing.Union[str, typing.List[str]], 
        ppkt_store_version: typing.Optional[str] = None
        ) -> typing.List[ppkt.Phenopacket]:
        """
        Retrieve Phenopacket objects for a specific cohort from a Phenopacket Store.

        Args:
            cohort_name (Union[str, List[str]]): A cohort name or list of names
            ppkt_store_version (str):  a `str` with Phenopacket Store release tag (e.g. `0.1.23`) or `None`
            if the *latest* release should be loaded.

        Returns:
            a list of GA4GH Phenopacket objects for the cohort
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