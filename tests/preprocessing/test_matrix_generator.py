import pytest
import phenopackets as ppkt
import pandas as pd
from unittest.mock import patch
from ppkt2synergy import PhenopacketMatrixGenerator
import pathlib

TEST_DIR = pathlib.Path(__file__).parent.parent.resolve()
HP_JSON_FILE = TEST_DIR/"data/hp.json"

@pytest.fixture
def mock_phenopackets():
    return [
        ppkt.Phenopacket(
            id="Patient_1",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_2",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:603903", label="Ehlers-Danlos syndrome"))
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_3",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures"), excluded=True),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
            ]
        )
    ]

def test_hpo_and_disease_matrix( mock_phenopackets):

    matrix_generator = PhenopacketMatrixGenerator(mock_phenopackets, hpo_file=str(HP_JSON_FILE))

    hpo_matrix_no_propagation, hpo_labels_no = matrix_generator._generate_hpo_term_status_matrix(propagate_hierarchy=False)
    assert isinstance(hpo_matrix_no_propagation, pd.DataFrame)
    assert hpo_matrix_no_propagation.loc["Patient_1", "HP:0020219"] == 1
    assert pd.isna(hpo_matrix_no_propagation.loc["Patient_1", "HP:0001250"])

    hpo_matrix_with_propagation, hpo_labels = matrix_generator._generate_hpo_term_status_matrix(propagate_hierarchy=True)
    assert isinstance(hpo_matrix_with_propagation, pd.DataFrame)
    assert hpo_matrix_with_propagation.loc["Patient_1", "HP:0001250"] == 1

    disease_matrix, disease_labels = matrix_generator._generate_disease_status_matrix()
    assert isinstance(disease_matrix, pd.DataFrame)
    assert disease_matrix.loc["Patient_1", "OMIM:101600"] == 1
    assert pd.isna(disease_matrix.loc["Patient_1", "OMIM:603903"])

    assert hpo_labels["HP:0020219"] == "Motor seizure"
    assert disease_labels["OMIM:101600"] == "Marfan Syndrome"
