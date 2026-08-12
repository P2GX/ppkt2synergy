Build Dataset
=============

The ``PhenotypeDatasetBuilder`` converts raw phenopackets conforming to the `GA4GH Phenopacket schema <https://phenopacket-schema.readthedocs.io/en/latest/index.html>`_ into a structured representation that can be used for correlation and synergy analysis.


Core usage
----------

To build a dataset, you must first load your local phenopacket JSON files using standard 
Python utilities (e.g., ``pathlib`` and official ``phenopackets`` protobuf parsers), and 
then pass the resulting list into the builder.

.. code-block:: python

    from pathlib import Path
    from google.protobuf.json_format import Parse
    from phenopackets import Phenopacket
    from phenosign import PhenotypeDatasetBuilder

    # 1. Locate your local phenopacket directory
    phenopacket_dir = Path("path/to/your/fbn1_phenopackets/")

    # 2. Iterate and parse JSON files into formal Phenopacket objects
    phenopackets = []
    for file_path in phenopacket_dir.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            phenopacket = Parse(data, Phenopacket())
            phenopackets.append(phenopacket)

    # 3. Initialize the builder and construct the analysis-ready dataset
    builder = PhenotypeDatasetBuilder(phenopackets)
    dataset = builder.build(
        missing_threshold=0.9,
        build_gpsea_cohort=True,
    )


Dataset overview
^^^^^^^^^^^^^^^^

The resulting ``PhenotypeDataset`` contains three components:

- **hpo_data** — binary HPO feature matrix across individuals, with an optional
  term relationship mask.
- **phenopackets** — the original phenopacket objects, retained for reference
  and downstream computations.
- **gpsea_cohort** — a preprocessed GPSEA cohort object for variant-aware analyses
  (present only when ``build_gpsea_cohort=True``).

Two index properties provide convenient access to the matrix dimensions:

.. code-block:: python

    dataset.individual_ids   # pd.Index of subject identifiers (rows)
    dataset.feature_ids      # pd.Index of HPO term identifiers (columns)


Key parameters
^^^^^^^^^^^^^^

``missing_threshold``
  Controls which HPO terms are retained based on how often they are observed
  across individuals.

  ``missing_threshold=0.9`` keeps only HPO terms observed (or explicitly excluded)
  in at least 10% of individuals. Set to ``1.0`` to retain all terms regardless
  of missingness.

``build_gpsea_cohort``
  If ``True``, constructs a GPSEA-compatible cohort object and attaches it to
  ``dataset.gpsea_cohort``. Required for variant-based condition analysis.
  Defaults to ``True``.


Advanced usage
--------------

Customizing HPO configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the builder utilizes internal ontology references. You can explicitly override 
the HPO source file or pinpoint a specific historical release for reproducibility:

.. code-block:: python

    builder = PhenotypeDatasetBuilder(
        phenopackets,
        hpo_file="path/to/hp.json",
        hpo_release="2023-10-09",
    )


Next steps
----------

After constructing the dataset:

- See :doc:`correlation` to explore phenotype–phenotype relationships.
- See :doc:`synergy` to identify condition-dependent adaptive synergy interactions.