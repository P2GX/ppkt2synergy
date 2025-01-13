import pytest 
import phenopackets as ppkt
import typing


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
        'patient, hpo_id_A, hpo_id_B, result_tuple',
        [
            (phenopkt, shortStatureOc.id, microcephalyOc.id, (1,1)),
            (phenopkt, shortStatureOc.id, aniridiaOc.id, None),



        ]




    )
    def test_get_status(self,
                        patient:ppkt.Phenopacket,
                        hpo_id_A:str, 
                        hpo_id_B:str, 
                        result_tuple: typing.Union[None, typing.Tuple[int, int]]):
        assert True