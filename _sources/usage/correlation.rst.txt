Correlation Analysis
====================

Once a dataset has been constructed, pairwise associations between HPO terms
can be quantified using ``HPOCorrelationAnalyzer``.


What is correlation?
--------------------

Correlation measures the association between two phenotypes:

- **positive values** → phenotypes tend to co-occur  
- **negative values** → phenotypes tend to occur in different individuals  
- **values near zero** → little or no association  


Correlation methods
-------------------

The analyzer supports correlation measures specifically optimized for binary (presence/absence) phenotype data:

- ``PHI`` — Phi coefficient ($\phi$), which mathematically equivalent to Pearson correlation for two binary variables. It measures the strength and direction of linear association between HPO terms.
- ``FISHER`` — Fisher's Exact Test, which computes the exact hypergeometric probability of the contingency table. It provides highly robust p-values alongside odds ratios, making it the preferred method for testing non-random associations in low-frequency HPO terms.

For most clinical phenotypes, ``FISHER`` provides a reliable statistical significance cutoff, while ``PHI`` serves as an excellent standardized effect size for downstream network visualizations.


Core usage
----------

.. code-block:: python

    from phenosign import HPOCorrelationAnalyzer

    analyzer = HPOCorrelationAnalyzer(
        dataset=dataset,
        min_individuals_for_correlation_test=30,
    )

    results = analyzer.compute_correlation_matrix(
        n_jobs=-1,
        include_pmids=False,
    )
    results.results_table.head()

The main parameters control data filtering, correlation type, and parallelization.


Key parameters
^^^^^^^^^^^^^^

- ``min_individuals_for_correlation_test``  
  Minimum number of individuals required to evaluate a feature pair.  
  Higher values increase robustness, while lower values allow more pairs to be tested.

- ``n_jobs``  
  Number of parallel jobs for computing pairwise correlations.
  Set to ``-1`` to use all available CPU cores.

- `include_pmids``
  If ``True``, tracks and aggregates underlying PubMed IDs (PMIDs) contributing to the phenotypic overlaps for downstream publication verification.


.. warning::

   Correlation requires variation in the data — each HPO term must have both
   observed and excluded values across individuals. Terms with no variation
   are automatically skipped.

.. note::

   Ontologically related HPO terms may be excluded from pairwise testing to avoid
   spurious associations. Multiple-testing correction is applied automatically.


Save results
------------

.. code-block:: python

    results.save_correlation_results(
        corr_threshold=0.1,
        adj_pval_threshold=0.05,
        output_file="correlation_results.csv",
    )

Set ``include_pmids=True`` in ``compute_correlation_matrix`` to include associated
PMIDs in the saved output.


Visualization
-------------

.. code-block:: python

    results.plot_correlation_heatmap_with_significance(
        corr_threshold=0.1,
        adj_pval_threshold=0.05,
    )

``corr_threshold`` sets the minimum correlation strength; ``adj_pval_threshold``
controls statistical significance. Lower thresholds include more pairs;
higher thresholds focus on stronger and more reliable associations.

.. code-block:: python

    results.save_correlation_heatmap(
        output_file="correlation_heatmap.html",
    )

The output format is determined automatically from the file extension.

- ``.html`` saves an interactive Plotly figure with zooming, panning,
  and hover tooltips.
- ``.svg``, ``.png``, ``.pdf``, and ``.jpg`` save static
  publication-ready figures.

For static image formats, the optional ``width``, ``height``, and
``scale`` parameters can be used to control the output resolution.

Next steps
----------

- See :doc:`synergy` for higher-order interaction analysis