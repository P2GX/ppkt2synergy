import pytest 
import phenopackets as ppkt
import typing
from ppkt2synergy import CohortStats
import pandas as pd
from google.protobuf.json_format import Parse
import os
import hpotk
import logging
HP_JSON_FILENAME = os.path.join(os.path.dirname(__file__),'data','hp.json')
print(hpotk.load_minimal_ontology(HP_JSON_FILENAME))

def load_phenopackets_from_json_folder(folder_path: str) -> typing.List[ppkt.Phenopacket]:
    """
    Load all JSON files from the specified folder and convert them into a list of Phenopacket objects.
    
    Args:
        folder_path: The path to the folder containing the JSON files.

    Returns: 
        A list of Phenopacket objects created from the JSON files.
    """
    phenopacket_list = []
    
    # Iterate through all the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  
            file_path = os.path.join(folder_path, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as jsf:
                phenopacket = Parse(message=ppkt.Phenopacket(), text=jsf.read())
                
                phenopacket_list.append(phenopacket)
    
    return phenopacket_list

@pytest.fixture
def phenopackets_DHCR24():
    folder_path = os.path.join(os.path.dirname(__file__), 'data', 'cohorts', 'DHCR24')
    return load_phenopackets_from_json_folder(folder_path)

class TestStats:

    @pytest.fixture
    def cohort_stats(self,phenopackets_DHCR24):
        cohort_name = "example_cohort_DHCR24"
        ppkt_store_version = "v1"
        
        stats = CohortStats(
            #cohort_name=cohort_name,
            #ppkt_store_version=ppkt_store_version,
            phenopackets=phenopackets_DHCR24,
            url=HP_JSON_FILENAME
            
        )
        #stats.phenopackets = phenopackets_DHCR24
        stats.hpo_term_observation_matrix = stats.generate_hpo_term_status_matrix()
        stats.hpo = hpotk.load_minimal_ontology(HP_JSON_FILENAME)
        return stats
    
    def test_generate_hpo_term_status_matrix(
        self,
        cohort_stats
    ):
        result = cohort_stats.generate_hpo_term_status_matrix()
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 10
        assert result.shape[1] == 29


    @pytest.mark.parametrize(
        'hpo_id_A, hpo_id_B, expected_exception, expected_shape',
        [
            ('HP:0000470', 'HP:0000369', None, (7, 2)), 
            ('HP:0001883', 'HP:0002007', None, (5, 2)),
            ('HP:0004322', 'HP:0002007', ValueError, None), # 'HP:0004322'has too much NaN 
            ('HP:0000470', 'HP:0001634', ValueError, None), # 'HP:0001634' does not exist in tests data
            ('HP:0000470', 'HP:0000470', ValueError, None), # Both terms are the same

        ]
    )
    def test_filter_hpo_terms_and_dropna(
        self, 
        cohort_stats, 
        hpo_id_A, 
        hpo_id_B, 
        expected_exception, 
        expected_shape
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                cohort_stats.filter_hpo_terms_and_dropna(hpo_id_A, hpo_id_B)
        else:
            filtered_matrix = cohort_stats.filter_hpo_terms_and_dropna(hpo_id_A, hpo_id_B)
            assert isinstance(filtered_matrix, pd.DataFrame)
            assert filtered_matrix.shape == expected_shape

    @pytest.mark.parametrize(
        'hpo_id, expected_exception, except_length',
        [
            ('HP:0000470', None, 25), 
            ('HP:0001883', None, 25),
            ('HP:0004322', ValueError, None), # 'HP:0004322'has too much NaN 
            ('HP:0001634', ValueError, None), # 'HP:0001634' does not exist in tests data

        ]
    )
    def test_find_unrelated_hpo_terms(
        self,
        cohort_stats,
        hpo_id,
        expected_exception,
        except_length
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                cohort_stats.find_unrelated_hpo_terms(hpo_id)
        else:
            unrelated_terms = cohort_stats.find_unrelated_hpo_terms(hpo_id)
            assert len(unrelated_terms) == except_length

    @pytest.mark.parametrize(
        "series_a, series_b, expected_exception, expected_stats",
        [
            (pd.Series([1, 0, 1, 1] * 10), pd.Series([0, 1, 1, 0] * 10), None, ["Spearman", "MCC"]), # Series with more than 30 elements
            (pd.Series([1, 1, 0, 1, 0] * 3), pd.Series([0, 0, 1, 1, 1] * 3), None, ["Fisher_exact"]), # Series with 15 elements
            (pd.Series([1, 0, 1, 1]), pd.Series([0, 1, 1, 0]), ValueError, None), # Series with less than 5 elements

        ]
    )
    def test_calculate_stats(
        self,
        cohort_stats, 
        series_a, 
        series_b, 
        expected_exception, 
        expected_stats
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                cohort_stats.calculate_stats(series_a, series_b)
        else:
            results = cohort_stats.calculate_stats(series_a, series_b)
            for stat in expected_stats:
                assert stat in results
                 

    @pytest.mark.parametrize(
    "hpo_id_A, hpo_id_B, expected_exception, expected_stats",
        [
            ("HP:0000252", "HP:0000256", None, ["Fisher_exact"]),
            ("HP:0001883", "HP:0002007", None, ["Fisher_exact"]),
            ("HP:0004322", "HP:0002007", ValueError, None), # 'HP:0004322'has too much NaN 
            ("HP:0000470", "HP:0001634", ValueError, None), # 'HP:0001634' does not exist in tests data
            ("HP:0000470", "HP:0000470", ValueError, None), #  Both terms are the same

        ]
    )
    def test_test_hpo_terms(
        self,
        cohort_stats, 
        hpo_id_A, 
        hpo_id_B, 
        expected_exception, 
        expected_stats
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                cohort_stats.test_hpo_terms(hpo_id_A, hpo_id_B)
        else:
            stats = cohort_stats.test_hpo_terms(hpo_id_A, hpo_id_B)
            for stat in expected_stats:
                assert stat in stats

    
   