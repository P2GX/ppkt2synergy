# phenosign

**phenosign** is a Python library for analyzing correlations and synergy in [GA4GH Phenopacket](https://www.ga4gh.org/product/phenopackets/) cohorts. 
---

## Installation

```bash
pip install phenosign
```


## Overview

This package enables the identification of pairwise associations and higher-order interactions between phenotypic features, helping to uncover biologically meaningful patterns in rare disease data.


## Features

* Correlation analysis of HPO features (Phi Coefficient)
* Synergy analysis to detect non-additive interactions between phenotypic features with respect to a target variable (e.g., variant effects or disease)
* Support for GA4GH phenopacket data
* Structured dataset construction from phenotypic profiles
* Visualization utilities (e.g., correlation heatmaps)


## Quickstart

```python
from pathlib import Path
import json
from phenosign import (
    PhenotypeDatasetBuilder,
    HPOCorrelationAnalyzer,
)

# Load phenopackets
phenopacket_dir = Path("path/to/your/fbn1_phenopackets/")

phenopackets = []
for file_path in phenopacket_dir.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data: str = f.read()
        phenopacket: Phenopacket = Parse(data, Phenopacket())
        phenopackets.append(phenopacket)

# Build dataset
dataset = PhenotypeDatasetBuilder(phenopackets).build()

# Run correlation analysis
analyzer = HPOCorrelationAnalyzer(dataset)
results = analyzer.compute_correlation_matrix()
results.result_table.head()
```

For a complete workflow and advanced options, see the [Documentation](https://phenosign.readthedocs.io/).



