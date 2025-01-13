import pytest 
import phenopackets as ppkt
import typing
from ppkt2synergy import get_status_for_terms


phenopkt = ppkt.Phenopacket()
shortStatureOc = ppkt.OntologyClass(id="HP:0004322", label="Short stature")
microcephalyOc = ppkt.OntologyClass(id="HP:0000252", label="Microcephaly")
deepPhiltrumOc = ppkt.OntologyClass(id="HP:0002002", label="Deep philtrum")
highPalateOc = ppkt.OntologyClass(id="HP:0000218", label="High palate")
shortStaturePf = ppkt.PhenotypicFeature(type=shortStatureOc)
microcephalyPf = ppkt.PhenotypicFeature(type=microcephalyOc)
phenopkt.phenotypic_features.append(shortStaturePf)
phenopkt.phenotypic_features.append(microcephalyPf)

# The following term was not added to the Phenopacket
aniridiaOc = ppkt.OntologyClass(id="HP:0000526", label="Aniridia")


class TestStats:

    @pytest.mark.parametrize(
        'phenopkt, hpo_id_A, hpo_id_B, expected_result',
        [
            (phenopkt, shortStatureOc.id, microcephalyOc.id, (1,1)),
            (phenopkt, shortStatureOc.id, aniridiaOc.id, None),



        ]




    )
    def test_get_status(self,
                        phenopkt: ppkt.Phenopacket,
                        hpo_id_A: str, 
                        hpo_id_B: str, 
                        expected_result: typing.Union[None, typing.Tuple[int, int]]):
        result = get_status_for_terms(phenopkt=phenopkt, hpo_id_A=hpo_id_A, hpo_id_B=hpo_id_B)
        assert expected_result == result