import pytest
from ppkt2synergy import CohortDataLoader
import phenopackets as ppkt

class TestCohortDataLoader:

    def test_from_ppkt_store(self):

        cohort_name = "FBN1"
        ppkt_store_version = "0.1.23"

        phenopackets = CohortDataLoader.from_ppkt_store(
            cohort_name=cohort_name,
            ppkt_store_version=ppkt_store_version,
        )
        
        assert isinstance(phenopackets, list)
        assert len(phenopackets) == 144  
        assert isinstance(phenopackets[0], ppkt.Phenopacket)