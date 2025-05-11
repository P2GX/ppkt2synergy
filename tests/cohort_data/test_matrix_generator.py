import pytest
import json
import phenopackets as ppkt
import pandas as pd
from unittest.mock import patch
from ppkt2synergy import PhenopacketMatrixGenerator
import pathlib

TEST_DIR = pathlib.Path(__file__).parent.parent.resolve()
HP_JSON_FILE = pathlib.Path(TEST_DIR, "data", "hp.json")

# Function to load HPO data from a file
@pytest.fixture
def load_hpo_from_file():
    """Load HPO terms hierarchy from a JSON file."""
    with open(HP_JSON_FILE, "r") as file:
        hpo_data = json.load(file)
    return hpo_data

@pytest.fixture
def mock_phenopackets():
    """
    Create a set of mock Phenopacket data, including HPO terms and disease diagnoses.
    This will be shared by multiple test cases.
    """
    return [
        ppkt.Phenopacket(
            id="Patient_1",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            interpretations=[
                ppkt.Interpretation(
                    diagnosis=ppkt.Diagnosis(disease=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
                )
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_2",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure"))
            ],
            interpretations=[
                ppkt.Interpretation(
                    diagnosis=ppkt.Diagnosis(disease=ppkt.OntologyClass(id="OMIM:603903", label="Ehlers-Danlos syndrome"))
                )
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_3",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures"), excluded=True),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            interpretations=[
                ppkt.Interpretation(
                    diagnosis=ppkt.Diagnosis(disease=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
                )
            ]
        )
    ]

@patch("ppkt2synergy.cohort_data.load_hpo")
def test_hpo_and_disease_matrix(mock_load_hpo, mock_phenopackets,load_hpo_from_file):
    """Test the generation of the HPO term status matrix and disease status matrix with and without hierarchy propagation."""

    mock_load_hpo.return_value = load_hpo_from_file  # Provide the HPO hierarchy to the generator
    
    # Instantiate the matrix generator without passing propagate_hierarchy until calling generate_hpo_term_status_matrix
    matrix_generator = PhenopacketMatrixGenerator(mock_phenopackets, hpo_file="tests/data/hp.json")

#                 HP:0020219   HP:0001250   HP:0012759
    #Patient 1        1            NaN            1
    #Patient 2        1            1            NaN
    #Patient 3        NaN          0              1
    # Generate the HPO matrix (without propagation)
    hpo_matrix_without_propagation, _ = matrix_generator.generate_hpo_term_status_matrix(propagate_hierarchy=False)

    #-HP:0001250
    #   - HP:0020219 
    #                 HP:0020219   HP:0001250   HP:0012759
    #Patient 1        1            1              1
    #Patient 2        1            1            NaN
    #Patient 3        0            0              1
    # Generate the HPO matrix (without propagation)
    # Generate the HPO matrix (with propagation)
    hpo_matrix_with_propagation, hpo_labels = matrix_generator.generate_hpo_term_status_matrix(propagate_hierarchy=True)

    # Generate the disease matrix
    disease_matrix, disease_labels = matrix_generator.generate_target_status_matrix()

    # Assertions for HPO matrices (with and without hierarchy propagation)
    assert isinstance(hpo_matrix_with_propagation, pd.DataFrame)
    assert isinstance(hpo_matrix_without_propagation, pd.DataFrame)

    # Assertion for disease matrix
    assert isinstance(disease_matrix, pd.DataFrame)

    # Check the behavior of HPO terms without propagation
    assert hpo_matrix_without_propagation.loc["Patient_1", "HP:0020219"] == 1
    assert pd.isna(hpo_matrix_without_propagation.loc["Patient_1", "HP:0001250"])
    assert hpo_matrix_without_propagation.loc["Patient_1", "HP:0012759"] == 1 
    assert hpo_matrix_without_propagation.loc["Patient_2", "HP:0020219"] == 1
    assert hpo_matrix_without_propagation.loc["Patient_2", "HP:0001250"] == 1
    assert pd.isna(hpo_matrix_without_propagation.loc["Patient_2", "HP:0012759"])  # NaN because no data
    assert pd.isna(hpo_matrix_without_propagation.loc["Patient_3", "HP:0020219"])
    assert hpo_matrix_without_propagation.loc["Patient_3", "HP:0012759"] == 1 
    assert hpo_matrix_with_propagation.loc["Patient_3", "HP:0001250"] == 0

    # Check the behavior of HPO terms with propagation
    assert hpo_matrix_with_propagation.loc["Patient_1", "HP:0020219"] == 1
    assert hpo_matrix_with_propagation.loc["Patient_1", "HP:0001250"] == 1  # Propagates to ancestors node
    assert hpo_matrix_with_propagation.loc["Patient_1", "HP:0012759"] == 1  
    assert hpo_matrix_with_propagation.loc["Patient_2", "HP:0020219"] == 1
    assert hpo_matrix_with_propagation.loc["Patient_2", "HP:0001250"] == 1
    assert hpo_matrix_with_propagation.loc["Patient_3", "HP:0012759"] == 1
    assert hpo_matrix_with_propagation.loc["Patient_3", "HP:0020219"] == 0  # Propagates to descendants node
    assert hpo_matrix_with_propagation.loc["Patient_3", "HP:0001250"] == 0

    # Check the disease matrix content
    assert disease_matrix.loc["Patient_1", "OMIM:101600"] == 1  # Marfan Syndrome
    assert pd.isna(disease_matrix.loc["Patient_1", "OMIM:603903"])  # Ehlers-Danlos syndrome
    assert pd.isna(disease_matrix.loc["Patient_2", "OMIM:101600"])  # Marfan Syndrome
    assert disease_matrix.loc["Patient_2", "OMIM:603903"] == 1  # Ehlers-Danlos syndrome
    assert disease_matrix.loc["Patient_3", "OMIM:101600"] == 1  # Marfan Syndrome
    assert pd.isna(disease_matrix.loc["Patient_3", "OMIM:603903"])  # Ehlers-Danlos syndrome

    # Check the disease labels
    assert disease_labels["OMIM:101600"] == "Marfan Syndrome"
    assert disease_labels["OMIM:603903"] == "Ehlers-Danlos syndrome"

    # Check the HPO labels
    assert hpo_labels["HP:0020219"] == "Motor seizure"
    assert hpo_labels["HP:0001250"] == "Seizures"
    assert hpo_labels["HP:0012759"] == "Neurological abnormality"
