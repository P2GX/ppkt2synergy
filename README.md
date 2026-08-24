# phenosign

**phenosign** is a Python library for analyzing pairwise relationships between Human Phenotype Ontology (HPO) features in [GA4GH Phenopacket](https://www.ga4gh.org/product/phenopackets/) cohorts.

It provides two complementary analyses:

1. **Pairwise association analysis**, which measures associations between binary phenotype annotations using the phi coefficient and Fisher's exact test.
2. **Target-specific synergy analysis**, which evaluates whether a pair of phenotype features provides joint information about a predefined target beyond the information provided by the individual features.

---

## Installation

```bash
pip install phenosign
```


## Features

* Construction of phenotype matrices from GA4GH Phenopackets, with separate representation of observed, excluded, and unreported HPO annotations
* Ontology-aware propagation of phenotype annotations
* Pairwise phenotype association analysis using the phi coefficient, Fisher's exact test, and Benjamini–Hochberg correction, with exclusion of ancestor–descendant HPO term pairs
* Mutual-information-based phenotype-pair synergy analysis with permutation testing and Benjamini–Hochberg correction, supporting targets such as disease diagnosis, variant effect, and sex
* Tabular results and interactive heatmaps, including contingency counts, effective sample sizes, adjusted p-values, and publication provenance where available


## Quickstart

```python
from pathlib import Path
from google.protobuf.json_format import Parse
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
dataset = PhenotypeDatasetBuilder(phenopackets).build(build_gpsea_cohort=False)

# Run correlation analysis
analyzer = HPOCorrelationAnalyzer(dataset)
results = analyzer.compute_correlation_matrix()
results.result_table.head()
```

For complete workflows, synergy analysis, visualization options, and API details, see the [Documentation](https://phenosign.readthedocs.io/).



