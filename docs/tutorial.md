# Tutorial

ppkt2synergy allws users to perform both correlation and synergy analyis (TODO explanations!).
We recommend the use of Jupyter notebooks to perform the analysis, but one could also use a Python script.

## Correlation analysis

First, we import the necessary packages. (Explain phenopacket store). The name of the cohort should be the same as that used in phenopacket store. 
```python
from ppkt2synergy import CohortDataLoader, HPOStatisticsAnalyzer, PhenopacketMatrixProcessor, CorrelationType
from ppkt2synergy import __version__
cohort_name = "CTCF"
phenopackets = CohortDataLoader.from_ppkt_store(cohort_name=cohort_name)
print(f"[ppkt2synergy] version {__version__} loaded")
print(f"n={len(phenopackets)} phenopackets from {cohort_name} cohort.")
```

## Ingest the HPO data

The library creates a matrix of HPO co-occurences.

```python
hpo_matrix, _ = PhenopacketMatrixProcessor.prepare_hpo_data(
    phenopackets=phenopackets, 
    threshold=0, 
    mode=None, 
    use_label=True,
    nan_strategy=None)
```