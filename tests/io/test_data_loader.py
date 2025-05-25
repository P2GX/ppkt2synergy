import pytest
from ppkt2synergy import CohortDataLoader
import phenopackets as ppkt


def test_from_ppkt_store():
    """
    Test the CohortDataLoader.from_ppkt_store method to ensure it correctly loads Phenopackets data.

    This test verifies:
    1. The returned data type is a list.
    2. The list contains 144 Phenopackets, which is the expected number for the FBN1 cohort.
    3. The first element in the list is an instance of ppkt.Phenopacket.
    """
    cohort_name = "FBN1"
    ppkt_store_version = "0.1.23"

    phenopackets = CohortDataLoader.from_ppkt_store(
        cohort_name=cohort_name,
        ppkt_store_version=ppkt_store_version,
    )

    assert isinstance(phenopackets, list)
    assert len(phenopackets) == 144
    assert isinstance(phenopackets[0], ppkt.Phenopacket)