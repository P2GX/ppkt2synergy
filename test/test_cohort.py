import pytest
from ppkt2synergy import CohortManager
import phenopackets as ppkt

class TestCohortManager:

    def test_from_ppkt_store(self):

        cohort_name = "FBN1"
        ppkt_store_version = "0.1.23"

        phenopackets = CohortManager.from_ppkt_store(
            cohort_name=cohort_name,
            ppkt_store_version=ppkt_store_version,
        )
        
        assert isinstance(phenopackets, list)
        assert len(phenopackets) == 144  
        assert isinstance(phenopackets[0], ppkt.Phenopacket)

    