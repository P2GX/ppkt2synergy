from dataclasses import dataclass, field
from collections.abc import Callable
import logging

import numpy as np
import pandas as pd
from gpsea.model import Patient, Cohort
import phenopackets as ppkt

from .features_data import HpoFeatureData

logger = logging.getLogger(__name__)


@dataclass
class PhenotypeDataset:
    """
    High-level dataset wrapper for phenotype-based analysis.

    This class integrates HPO feature matrices, phenopacket metadata,
    and optional GPSEA variant annotations into a unified analysis
    interface.
    """

    hpo_data: HpoFeatureData
    phenopackets: list[ppkt.Phenopacket] = field(default_factory=list)
    gpsea_cohort: Cohort | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hpo_data, HpoFeatureData):
            raise TypeError("`hpo_data` must be an HpoFeatureData instance.")
        
    @property
    def individual_ids(self) -> pd.Index:
        """Individual IDs (matrix index)."""
        return self.hpo_data.individual_ids

    @property
    def feature_ids(self) -> pd.Index:
        """HPO feature IDs (matrix columns)."""
        return self.hpo_data.feature_ids
    
    def describe_conditions(self) -> pd.DataFrame:
        """
        Generate a unified cohort summary table.

        The table combines disease, sex, gene, and variant-effect summaries
        into a long-format DataFrame.

        Returns
        -------
        pd.DataFrame
            A table with the following columns:

            - category :
                Type of cohort characteristic, such as ``Disease``, ``Sex``,
                ``Gene``, or ``Variant effect``.

            - item :
                Specific value within the category.

            - n_individuals :
                Number and percentage of individuals represented by the item.
        """
        tables: list[pd.DataFrame] = []

        diseases_df = self.list_diseases().reset_index()

        if not diseases_df.empty:
            diseases_df["item"] = diseases_df.apply(
                lambda row: (
                    f"{row['label']} [{row['disease_id']}]"
                    if row["label"] != "Unknown Label"
                    else row["disease_id"]
                ),
                axis=1,
            )
            diseases_df["category"] = "Disease"

            tables.append(
                diseases_df[
                    ["category", "item", "n_individuals"]
                ]
            )

        sex_df = self.describe_sex().reset_index()

        if not sex_df.empty:
            sex_df = sex_df.rename(columns={"sex": "item"})
            sex_df["category"] = "Sex"

            tables.append(
                sex_df[
                    ["category", "item", "n_individuals"]
                ]
            )

        genes_df = self.list_genes().reset_index()

        if not genes_df.empty:
            genes_df = genes_df.rename(columns={"gene_symbol": "item"})
            genes_df["category"] = "Gene"

            tables.append(
                genes_df[
                    ["category", "item", "n_individuals"]
                ]
            )

        if self.gpsea_cohort is not None:
            variant_effects_df = self.variant_effect_summary()

            if not variant_effects_df.empty:
                variant_effects_df = (
                    variant_effects_df
                    .rename_axis(index="variant_effect")
                    .reset_index()
                    .melt(
                        id_vars="variant_effect",
                        var_name="transcript_id",
                        value_name="n_individuals",
                    )
                )

                variant_effects_df["item"] = (
                    variant_effects_df["variant_effect"].astype(str)
                    + " — "
                    + variant_effects_df["transcript_id"].astype(str)
                )
                variant_effects_df["category"] = "Variant effect"

                tables.append(
                    variant_effects_df[
                        ["category", "item", "n_individuals"]
                    ]
                )

        if not tables:
            return pd.DataFrame(
                columns=["category", "item", "n_individuals"]
            )

        return pd.concat(
            tables,
            ignore_index=True,
        )
    
        for patient in self.gpsea_cohort.all_patients:
            for variant in patient.variants:
                for txa in variant.tx_annotations:
                    print(type(txa))
                    print([
                        attr
                        for attr in dir(txa)
                        if not attr.startswith("_")
                    ])
                    raise RuntimeError("Stop after first transcript annotation")
        
    def get_condition(
        self,
        predicate: Callable[[ppkt.Phenopacket], bool | None],
        *,
        name: str | None = None,
    ) -> pd.Series:
        """
        Convert a phenopacket predicate into an individual-level binary condition.

        Parameters
        ----------
        predicate : Callable[[ppkt.Phenopacket], bool | None]
            Function that maps a phenopacket to True, False, or None (unknown).

        name : str | None, optional
            Name of the condition to store in cache. Default is None.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values:
            - 1.0 = True
            - 0.0 = False
            - NaN = unknown

        Raises
        ------
        TypeError
            If predicate returns a non-bool/non-None value.

        RuntimeError
            If predicate raises an exception for a specific individual.
        """
        results: dict[str, float] = {}

        for phenopacket in self.phenopackets:
            pid = str(phenopacket.id)
            try:
                value = predicate(phenopacket)
            except Exception as exc:
                raise RuntimeError(
                    f"Condition predicate failed for "
                    f"individual {pid!r}."
                ) from exc

            if value is None:
                results[pid] = np.nan

            elif isinstance(value, (bool, np.bool_)):
                results[pid] = float(value)

            else:
                raise TypeError(
                    f"Condition predicate for individual {pid!r} must return bool or None, "
                    f"got {type(value).__name__}."
                )

        return pd.Series(
            results,
            dtype="float64",
            name=name,
        ).reindex(self.individual_ids)
            
    def get_variant_condition(
        self,
        predicate: Callable[[Patient], bool | None],
        *,
        name: str | None = None,
    ) -> pd.Series:
        """
        Build a binary condition vector for transcript-aware variant effects.

        Parameters
        ----------
        predicate : Callable[[Patient], bool | None]
            Function that maps a GPSEA Patient to True, False, or None.
            
        name : str | None, optional
            Name of the condition to store in cache. Default is None.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values:
            - 1.0 : True
            - 0.0 : False
            - NaN : unknown

        Raises
        ------
        ValueError
            If GPSEA cohort is not available.
        """
        if self.gpsea_cohort is None:
            raise ValueError(
                "No GPSEA cohort available. Variant condition requires "
                "a preprocessed GPSEA cohort."
            )

        cohort_map = {str(p._labels.meta_label): p for p in self.gpsea_cohort.all_patients}
        results: dict[str, float] = {}

        for individual_id in self.individual_ids:
            pid = str(individual_id)
            patient = cohort_map.get(pid)

            if patient is None:
                results[pid] = np.nan
                continue

            value = predicate(patient)

            if value is None:
                results[pid] = np.nan
            else:
                results[pid] = float(value)

        return pd.Series(
            results,
            dtype="float64",
            name=name,
        ).reindex(self.individual_ids)

    def get_pmids(self) -> pd.Series:
        """
        Retrieve PubMed IDs associated with a list of individuals.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values: list of PMIDs as strings. Empty list if no PMIDs found.
        """
        results: dict[str, list[str]] = {}

        for phenopacket in self.phenopackets:
            pid = str(phenopacket.id)
            results[pid] = [] if phenopacket is None else self._extract_pmids(phenopacket)

        return pd.Series(
            results, 
            name="pmids", 
            dtype="object"
        ).reindex(self.individual_ids)
    
    @staticmethod
    def _extract_pmids(phenopacket: ppkt.Phenopacket) -> list[str]:
        """
        Extract PubMed IDs from a phenopacket's metadata.

        Parameters
        ----------
        phenopacket : ppkt.Phenopacket
            The phenopacket from which to extract PMIDs.  

        Returns
        -------
        list[str]
            A list of extracted PubMed IDs.
        """ 
        pmids: list[str] = []

        meta_data = getattr(phenopacket, "meta_data", None)
        if meta_data is None:
            return pmids

        for ref in getattr(meta_data, "external_references", []):
            ref_id = getattr(ref, "id", None)
            if isinstance(ref_id, str) and ref_id.startswith("PMID:"):
                pmids.append(ref_id.removeprefix("PMID:"))

        return sorted(set(pmids)) 

    def list_diseases(self) -> pd.DataFrame:
        """
        Summarize observed diseases across individuals.

        Returns
        -------
        pd.DataFrame

            Columns:

            - disease_id :
                Disease identifier (e.g., OMIM or ORPHA ID).

            - label :
                Human-readable disease label.

            - n_individuals :
                Number and percentage of individuals with the disease.
        """
        total_individuals = len(self.phenopackets)

        disease_labels: dict[str, str] = {}
        observed_counts: dict[str, int] = {}

        for phenopacket in self.phenopackets:
            seen_observed_in_patient: set[str] = set()

            for disease in getattr(phenopacket, "diseases", []):
                term = getattr(disease, "term", None)
                disease_id = getattr(term, "id", None)

                if not disease_id:
                    continue

                if bool(getattr(disease, "excluded", False)):
                    continue

                if disease_id not in disease_labels:
                    disease_labels[disease_id] = getattr(term, "label", None) or "Unknown Label"

                seen_observed_in_patient.add(disease_id)

            for disease_id in seen_observed_in_patient:
                observed_counts[disease_id] = observed_counts.get(disease_id, 0) + 1

        df = pd.DataFrame(
            [
                {
                    "disease_id": dis_id, 
                    "label": disease_labels[dis_id], 
                    "observed_raw": count
                }
                for dis_id, count in observed_counts.items()
            ]
        )

        if df.empty:
            df = pd.DataFrame(columns=["disease_id", "label", "n_individuals"])
        else:
            df = df.sort_values(
                ["observed_raw", "disease_id"],
                ascending=[False, True]
            ).reset_index(drop=True)

            denom = total_individuals if total_individuals > 0 else 1
            percentage = (df["observed_raw"] / denom) * 100

            df["n_individuals"] = (
                df["observed_raw"].astype(str) +
                " (" + percentage.round(1).astype(str) + "%)"
            )

            df = df.drop(columns=["observed_raw"])
        return df.set_index("disease_id")

    def describe_sex(self) -> pd.DataFrame:
        """
        Summarize sex distribution across individuals.

        Returns
        -------
        pd.DataFrame

            Columns:

            - sex :
                One of ``female``, ``male``, or ``unknown``.

            - n_individuals :
                Count of individuals with this sex.
        """
        sex_map = {
            1: "female",
            2: "male",
        }

        counts: dict[str, int] = {
            "female": 0,
            "male": 0,
            "unknown": 0,
        }

        for phenopacket in self.phenopackets:
            subject = getattr(phenopacket, "subject", None)
            if subject is None:
                counts["unknown"] += 1
                continue

            sex = sex_map.get(getattr(subject, "sex", None), "unknown")
            counts[sex] = counts.get(sex, 0) + 1

        total_individuals = len(self.phenopackets)

        df = pd.DataFrame(
            [
                {"sex": sex, "n_individuals_raw": n} 
                for sex, n in counts.items()
            ]
        )

        denom = total_individuals if total_individuals > 0 else 1
        percentage = (df["n_individuals_raw"] / denom) * 100

        df["n_individuals"] = (
            df["n_individuals_raw"].astype(str) + 
            " (" + percentage.round(1).astype(str) + "%)"
        )

        df = df.drop(columns=["n_individuals_raw"])

        return df.set_index("sex")

    def list_genes(self) -> pd.DataFrame:
        """
        List gene symbols annotated in the cohort.

        Returns
        -------
        pd.DataFrame

            Columns:

            - gene_symbol :
                HGNC gene symbol.

            - n_individuals :
                Number and percentage of individuals carrying variants in the gene.
        """
        total_individuals = len(self.phenopackets)

        counts: dict[str, int] = {}

        for phenopacket in self.phenopackets:
            unique_genes = set(self._extract_gene_symbols(phenopacket))
            for gene_symbol in unique_genes:
                counts[gene_symbol] = counts.get(gene_symbol, 0) + 1

        df = pd.DataFrame(
            [
                {"gene_symbol": gene_symbol, "n_individuals_raw": n}
                for gene_symbol, n in counts.items()
            ]
        )

        if df.empty:
            df = pd.DataFrame(columns=["gene_symbol", "n_individuals"])
        else:
            df = df.sort_values(
                ["n_individuals_raw", "gene_symbol"],
                ascending=[False, True],
            ).reset_index(drop=True)

            denom = total_individuals if total_individuals > 0 else 1
            percentage = (df["n_individuals_raw"] / denom) * 100

            df["n_individuals"] = (
                df["n_individuals_raw"].astype(str) + " (" + percentage.round(1).astype(str) + "%)"
            )

            df = df.drop(columns=["n_individuals_raw"])

        return df.set_index("gene_symbol")
    
    @staticmethod
    def _extract_gene_symbols(phenopacket: ppkt.Phenopacket) -> set[str]:
        """
        Extract gene symbols from a phenopacket's genomic interpretations.

        Parameters
        ----------
        phenopacket : ppkt.Phenopacket
            The phenopacket from which to extract gene symbols.

        Returns
        -------
        set[str]
            A set of unique gene symbols extracted from the phenopacket.
        """
        genes: set[str] = set()

        for interpretation in getattr(phenopacket, "interpretations", []):
            diagnosis = getattr(interpretation, "diagnosis", None)
            if diagnosis is None:
                continue

            for genomic_interpretation in getattr(
                diagnosis,
                "genomic_interpretations",
                [],
            ):
                gene_descriptor = getattr(genomic_interpretation, "gene_descriptor", None)
                if gene_descriptor is not None:
                    symbol = getattr(gene_descriptor, "symbol", None)
                    if symbol:
                        genes.add(symbol)

                variant_interpretation = getattr(
                    genomic_interpretation,
                    "variant_interpretation",
                    None,
                )
                if variant_interpretation is None:
                    continue

                variation_descriptor = getattr(
                    variant_interpretation,
                    "variation_descriptor",
                    None,
                )
                if variation_descriptor is None:
                    continue

                gene_context = getattr(variation_descriptor, "gene_context", None)

                if gene_context is not None:
                    symbol = getattr(gene_context, "symbol", None)
                    if symbol:
                        genes.add(symbol)

        return genes
    
    def variant_effect_summary(self) -> pd.DataFrame:
        """
        Summarize GPSEA variant effects by transcript.

        Calculates variant effect distributions for each transcript.
        Returns a transposed matrix where each cell contains both
        absolute counts and percentages.

        Returns
        -------
        pd.DataFrame

            Rows:
                Variant effect types (e.g., ``MISSENSE``, ``NONSENSE``)

            Columns:
                Transcript IDs

            Values:
                Strings formatted as ``"count (percentage%)"``

        Example:
            ``15 (75.0%)``

        Raises
        ------
        ValueError
            If no GPSEA cohort has been loaded.
        """
        if self.gpsea_cohort is None:
            raise ValueError("No GPSEA cohort available.")

        counters = self.gpsea_cohort.variant_effect_count_by_tx()

        preferred_tx_ids = set()
        for patient in self.gpsea_cohort.all_patients:
            for variant in patient.variants:
                for txa in variant.tx_annotations:
                    if getattr(txa, "is_preferred", False) and getattr(txa, "transcript_id", None):
                        preferred_tx_ids.add(txa.transcript_id)
        
        counters = {tx_id: counters[tx_id] for tx_id in preferred_tx_ids if tx_id in counters}

        raw_df = (
            pd.DataFrame.from_dict(counters, orient="index")
            .fillna(0)
            .astype(int)
        )

        if not raw_df.empty:
            raw_df = raw_df.sort_index()
            
            total_per_tx = raw_df.sum(axis=1)
            denom = total_per_tx.replace(0, 1)
            percentage_df = raw_df.divide(denom, axis=0) * 100
            
            summary = raw_df.astype(str) + " (" + percentage_df.round(1).astype(str) + "%)"
            
            for col in summary.columns:
                summary.loc[total_per_tx == 0, col] = "0 (0.0%)"
            
            summary = summary.T
            
            summary.index.name = "variant_effect"
            summary.columns.name = "transcript_id"
        else:
            summary = pd.DataFrame()

        return summary
    