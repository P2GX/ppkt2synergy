Synergy Analysis
================

After constructing a dataset, the ``SynergyAnalyzer`` identifies pairs of HPO features
whose joint effect on a condition cannot be explained by individual features alone.


What is synergy?
----------------

Synergy measures whether a pair of HPO features provides additional information
about a condition compared to each feature individually:

- **positive synergy** → the combination is more informative than each feature alone
- **near zero** → features contribute independently
- **negative synergy** → features are redundant with respect to the condition

Synergy is computed using mutual information with permutation-based significance testing.


Adaptive Permutation Testing (Early-Stopping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To optimize massive parallel computations across thousands of HPO pairs, **phenosign** implements an adaptive permutation framework:

- **Weak signals are terminated early** once the accumulated random shuffles cross a specific hit threshold (``target_successes``), preventing the CPU from wasting cycles on uninformative associations.
- **Strong synergistic signals automatically scale up** to deeper permutation ceilings (``max_perms``) to provide ultra-high statistical resolution, ensuring survival during strict multi-test False Discovery Rate (FDR) corrections.


Inspect the dataset
-------------------

Before defining a condition, inspect the cohort to understand available diseases,
genes, sex distribution, and variant effects:

.. code-block:: python

    dataset.describe_conditions()

``variant_effects_df`` is ``None`` if no GPSEA cohort was built during dataset construction.


Core usage
----------

.. code-block:: python

    from phenosign import SynergyAnalyzer

    synergy_analyzer = SynergyAnalyzer(
        dataset=dataset,
        min_individuals_for_synergy_calculation=40,
        random_state=42,
    )

Key parameters
^^^^^^^^^^^^^^

``min_individuals_for_synergy_calculation``
  Minimum number of individuals required to evaluate a feature pair.
  Higher values increase robustness; lower values allow more pairs to be tested.

``random_state``
  Seed for reproducible permutation testing.


Defining a condition
--------------------

Synergy analysis requires a **condition** — a binary vector indicating which
individuals belong to the positive group (1) and which do not (0).

Conditions are constructed by passing a **predicate function** to the dataset.
**phenosign** provides built-in helper functions to generate common predicates.

.. note::

   A condition must have both positive (1) and negative (0) samples present.
   Conditions with only a single class cannot be used for synergy analysis.

.. note::

   The built-in helpers cover common use cases, but you can define any predicate
   as a plain Python function. A phenopacket-level predicate must accept a
   ``Phenopacket`` and return ``True``, ``False``, or ``None`` (unknown):

   .. code-block:: python

      def my_predicate(phenopacket) -> bool | None:
          # your custom logic here
          return True

      condition = dataset.get_condition(my_predicate, name="my_condition")

   For variant-level predicates, the function receives a GPSEA ``Patient`` object
   instead. See the `GPSEA documentation <https://gpsea.readthedocs.io>`_ for details
   on the ``Patient`` data model.


Using built-in helpers
^^^^^^^^^^^^^^^^^^^^^^

The following helpers work at the phenopacket level and are passed to
``dataset.get_condition()``:

.. code-block:: python

    from phenosign import has_disease, has_sex, has_gene

    # By disease
    condition = dataset.get_condition(
        has_disease("OMIM:154700"),
        name="disease:Marfan syndrome",
    )

    # By sex
    condition = dataset.get_condition(
        has_sex("female"),
        name="sex:female",
    )

    # By gene
    condition = dataset.get_condition(
        has_gene("FBN1"),
        name="gene:FBN1",
    )

The ``name`` parameter is optional but recommended — it labels the condition
and enables caching for repeated queries.


Variant-based conditions
^^^^^^^^^^^^^^^^^^^^^^^^

For variant-level conditions, use ``dataset.get_variant_condition()`` with
helpers that operate on GPSEA ``Patient`` objects.

.. warning::

   Variant-based conditions require a GPSEA cohort.
   Set ``build_gpsea_cohort=True`` when constructing the dataset.

Filter by variant effect on a specific transcript:

.. code-block:: python

    from gpsea.model import VariantEffect
    from phenosign import has_variant_effect

    condition = dataset.get_variant_condition(
        has_variant_effect(
            transcript_id="NM_000138.5",
            variant_effect=VariantEffect.MISSENSE_VARIANT,
        ),
        condition_name="variant:missense_NM_000138.5",
    )

Filter by variant effect restricted to a specific exon:

.. code-block:: python

    from phenosign import has_exon_and_variant_effect

    condition = dataset.get_variant_condition(
        has_exon_and_variant_effect(
            transcript_id="NM_000138.5",
            exon=25,
            variant_effect=VariantEffect.MISSENSE_VARIANT,
        ),
        condition_name="variant:missense_exon25_NM_000138.5",
    )

.. warning::

   Synergy analysis requires variability in both conditions and HPO term pairs.
   Each condition must split individuals into at least two groups - if a
   condition has no variation, no synergy will be computed. Each HPO term
   pair must also have variability across individuals; pairs with no variation
   are automatically skipped and will not appear in the results.


Running the analysis
--------------------

Pass a condition to ``compute_synergy_matrix()``:

.. code-block:: python

    synergy_results = synergy_analyzer.compute_synergy_matrix(
        condition=condition,
        n_jobs=-1,
        n_perms = 5000,

    )

    synergy_results.results_table.head()


Adaptive parameters tuning guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``n_perms`` (default: 5000)
  The number of permutations used to estimate p-values. For small-cohort or rare disease cohorts, raising this to ``5000`` is highly recommended to suppress initial sampling noise.

``n_jobs`` (default: -1)
  Number of parallel workers. ``-1`` utilizes all available CPU cores.


Save results
------------

.. code-block:: python

    synergy_results.save_synergy_results(
        synergy_threshold=0.01,
        adj_pval_threshold=0.05,
        output_file="synergy_results.csv",
    )


Visualization
-------------

.. code-block:: python

    synergy_results.plot_synergy_heatmap(
        synergy_threshold=0.01,
        adj_pval_threshold=0.05,
        condition_name="disease:Marfan syndrome",
    )

``synergy_threshold`` specifies the minimum absolute synergy value
(``|synergy|``) required for a phenotype pair to be displayed.;
``adj_pval_threshold`` controls statistical significance.
Only phenotype pairs satisfying both criteria are included in the
heatmap. Lower thresholds retain more phenotype pairs, whereas higher
thresholds emphasize the strongest and most statistically robust
synergy- and redundancy-dominated interactions.

.. code-block:: python

    synergy_results.save_synergy_heatmap(
    output_file="synergy_heatmap.html"
    )

The output format is determined automatically from the file extension.

- ``.html`` saves an interactive Plotly figure with zooming, panning,
  and hover tooltips.
- ``.svg``, ``.png``, ``.pdf``, and ``.jpg`` save static
  publication-ready figures.

For static image formats, the optional ``width``, ``height``, and
``scale`` parameters can be used to control the output resolution.

.. note::

   Ontologically related HPO terms may be excluded from pairwise testing.
   Multiple-testing correction is applied automatically.


Next steps
----------

Synergy analysis can be combined with correlation analysis to distinguish:

- general phenotype associations (correlation)
- condition-dependent interactions (synergy)

See :doc:`correlation` to explore phenotype–phenotype relationships independently of a condition.