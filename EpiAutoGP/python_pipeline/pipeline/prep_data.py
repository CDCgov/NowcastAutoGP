import datetime as dt
import json
import logging
import os
import subprocess
import tempfile
from datetime import date
from logging import Logger
from pathlib import Path
from typing import Optional

import forecasttools
import jax.numpy as jnp
import polars as pl
import polars.selectors as cs

from pyrenew_hew.utils import approx_lognorm

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}


def get_nhsn(
    start_date: date,
    end_date: date,
    disease: str,
    loc_abb: str,
    temp_dir: Optional[Path] = None,
    credentials_dict: Optional[dict] = None,
    local_data_file: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Retrieve hospital admissions data from the National Healthcare Safety Network (NHSN).
    
    Downloads hospital admissions data for a specific disease, location, and date range
    using the CDC data.gov API through R's forecasttools package. Handles authentication
    and data format standardization.

    Parameters
    ----------
    start_date : date
        Beginning date for data retrieval (inclusive)
    end_date : date
        End date for data retrieval (inclusive)
    disease : str
        Disease name, must be one of: "COVID-19", "Influenza", "RSV"
    loc_abb : str
        Location abbreviation (e.g., "CA", "TX", "US" for national)
    temp_dir : Path, optional
        Directory for temporary files. If None, creates a temporary directory
    credentials_dict : dict, optional
        Dictionary containing API credentials with keys:
        - "nhsn_api_key_id": CDC API key ID
        - "nhsn_api_key_secret": CDC API key secret
        If None, uses environment variables NHSN_API_KEY_ID and NHSN_API_KEY_SECRET
    local_data_file : Path, optional
        Path to local parquet file to read instead of API call. If provided,
        skips the API download and reads from this file directly

    Returns
    -------
    pl.DataFrame
        Hospital admissions data with columns:
        - weekendingdate: Date (end of epidemiological week)
        - jurisdiction: Location abbreviation
        - hospital_admissions: Number of confirmed admissions (numeric)

    Raises
    ------
    RuntimeError
        If the R script execution fails or API call returns an error
    KeyError
        If the specified disease is not supported

    Notes
    -----
    - Requires R environment with forecasttools package installed
    - Uses CDC's NHSN hospital capacity and utilization dataset
    - Data represents confirmed new admissions by epidemiological week ending date
    - US national data uses jurisdiction "USA" in API but returns as "US"
    """
    if local_data_file is None:
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        if credentials_dict is None:
            credentials_dict = dict()

        def py_scalar_to_r_scalar(py_scalar):
            if py_scalar is None:
                return "NULL"
            return f"'{str(py_scalar)}'"

        disease_nhsn_key = {
            "COVID-19": "totalconfc19newadm",
            "Influenza": "totalconfflunewadm",
            "RSV": "totalconfrsvnewadm",
        }

        columns = disease_nhsn_key[disease]

        loc_abb_for_query = loc_abb if loc_abb != "US" else "USA"

        local_data_file = Path(temp_dir, "nhsn_temp.parquet")
        api_key_id = credentials_dict.get(
            "nhsn_api_key_id", os.getenv("NHSN_API_KEY_ID")
        )
        api_key_secret = credentials_dict.get(
            "nhsn_api_key_secret", os.getenv("NHSN_API_KEY_SECRET")
        )

        r_command = [
            "Rscript",
            "-e",
            f"""
            forecasttools::pull_data_cdc_gov_dataset(
                dataset = "nhsn_hrd_prelim",
                api_key_id = {py_scalar_to_r_scalar(api_key_id)},
                api_key_secret = {py_scalar_to_r_scalar(api_key_secret)},
                start_date = {py_scalar_to_r_scalar(start_date)},
                end_date = {py_scalar_to_r_scalar(end_date)},
                columns = {py_scalar_to_r_scalar(columns)},
                locations = {py_scalar_to_r_scalar(loc_abb_for_query)}
            ) |>
            dplyr::mutate(weekendingdate = as.Date(weekendingdate)) |>
            dplyr::mutate(jurisdiction = dplyr::if_else(jurisdiction == "USA", "US",
            jurisdiction
            )) |>
            dplyr::rename(hospital_admissions = {py_scalar_to_r_scalar(columns)}) |>
            dplyr::mutate(hospital_admissions = as.numeric(hospital_admissions)) |>
            forecasttools::write_tabular("{str(local_data_file)}")
            """,
        ]

        result = subprocess.run(r_command)

        if result.returncode != 0:
            raise RuntimeError(f"get_nhsn: {result.stderr.decode('utf-8')}")
    raw_dat = pl.read_parquet(local_data_file)
    dat = raw_dat.with_columns(weekendingdate=pl.col("weekendingdate").cast(pl.Date))
    return dat


def combine_surveillance_data(
    nssp_data: pl.DataFrame,
    nhsn_data: pl.DataFrame,
    disease: str,
    nwss_data: Optional[pl.DataFrame] = None,
):
    """
    Combine multiple surveillance data streams into a unified long-format dataset.
    
    Merges NSSP emergency department visits, NHSN hospital admissions, and optionally
    NWSS wastewater data into a single DataFrame with standardized column names and
    long-format structure suitable for epidemiological modeling.

    Parameters
    ----------
    nssp_data : pl.DataFrame
        NSSP emergency department visit data with columns including:
        - observed_ed_visits: Disease-specific ED visits
        - other_ed_visits: Other ED visits for denominator
        - date, geo_value, and other metadata columns
    nhsn_data : pl.DataFrame
        NHSN hospital admissions data with columns:
        - weekendingdate: Date (will be renamed to 'date')
        - jurisdiction: Location (will be renamed to 'geo_value') 
        - hospital_admissions: Confirmed admissions (will be renamed to 'observed_hospital_admissions')
    disease : str
        Disease name for metadata (e.g., "COVID-19", "Influenza", "RSV")
    nwss_data : pl.DataFrame, optional
        NWSS wastewater data with columns:
        - log_genome_copies_per_ml: Viral concentration (will be renamed to 'site_level_log_ww_conc')
        - location: Jurisdiction (will be renamed to 'geo_value')
        - lab_site_index: Lab-site combination index
        - Other wastewater-specific columns

    Returns
    -------
    pl.DataFrame
        Combined surveillance data in long format with columns:
        - date: Date of observation
        - geo_value: Geographic location identifier
        - .variable: Type of measurement (e.g., 'observed_ed_visits', 'observed_hospital_admissions')
        - .value: Numeric value of the measurement
        - lab_site_index: Lab-site index (only for wastewater data, None for others)
        - disease: Disease name
        - data_type: Always "train" for training data
        
    Notes
    -----
    - Converts all data sources to consistent long format for modeling
    - Harmonizes column names across different surveillance systems
    - Wastewater data includes lab_site_index for site-level modeling
    - ED visits and hospital admissions have lab_site_index set to None
    - All data is marked as "train" type for model fitting
    """
    nssp_data_long = nssp_data.unpivot(
        on=["observed_ed_visits", "other_ed_visits"],
        variable_name=".variable",
        index=cs.exclude(["observed_ed_visits", "other_ed_visits"]),
        value_name=".value",
    ).with_columns(pl.lit(None).alias("lab_site_index"))

    nhsn_data_long = (
        nhsn_data.rename(
            {
                "weekendingdate": "date",
                "jurisdiction": "geo_value",
                "hospital_admissions": "observed_hospital_admissions",
            }
        )
        .unpivot(
            on="observed_hospital_admissions",
            index=cs.exclude("observed_hospital_admissions"),
            variable_name=".variable",
            value_name=".value",
        )
        .with_columns(pl.lit(None).alias("lab_site_index"))
    )

    nwss_data_long = (
        nwss_data.rename(
            {
                "log_genome_copies_per_ml": "site_level_log_ww_conc",
                "location": "geo_value",
            }
        )
        .with_columns(pl.lit("train").alias("data_type"))
        .select(
            cs.exclude(
                [
                    "lab",
                    "log_lod",
                    "below_lod",
                    "site",
                    "site_index",
                    "site_pop",
                    "lab_site_name",
                ]
            )
        )
        .unpivot(
            on="site_level_log_ww_conc",
            index=cs.exclude("site_level_log_ww_conc"),
            variable_name=".variable",
            value_name=".value",
        )
        if nwss_data is not None
        else pl.DataFrame()
    )

    combined_dat = (
        pl.concat(
            [nssp_data_long, nhsn_data_long, nwss_data_long],
            how="diagonal_relaxed",
        )
        .with_columns(pl.lit(disease).alias("disease"))
        .sort(["date", "geo_value", ".variable"])
        .select(
            [
                "date",
                "geo_value",
                "disease",
                "data_type",
                ".variable",
                ".value",
                "lab_site_index",
            ]
        )
    )

    return combined_dat


def aggregate_to_national(
    data: pl.LazyFrame,
    geo_values_to_include: list[str],
    first_date_to_include: date,
    national_geo_value="US",
):
    assert national_geo_value not in geo_values_to_include
    return (
        data.filter(
            pl.col("geo_value").is_in(geo_values_to_include),
            pl.col("reference_date") >= first_date_to_include,
        )
        .group_by(["disease", "metric", "geo_type", "reference_date"])
        .agg(geo_value=pl.lit(national_geo_value), value=pl.col("value").sum())
    )


def process_loc_level_data(
    loc_level_nssp_data: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    first_training_date: date,
    loc_pop_df: pl.DataFrame,
) -> pl.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if loc_level_nssp_data is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "geo_value": pl.Utf8,
                "disease": pl.Utf8,
                "ed_visits": pl.Float64,
            }
        )

    disease_key = _disease_map.get(disease, disease)

    if loc_abb == "US":
        locations_to_aggregate = (
            loc_pop_df.filter(pl.col("abb") != "US")
            .get_column("abb")
            .unique()
            .to_list()
        )
        logger.info("Aggregating state-level data to national")
        loc_level_nssp_data = aggregate_to_national(
            loc_level_nssp_data,
            locations_to_aggregate,
            first_training_date,
            national_geo_value="US",
        )

    return (
        loc_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == loc_abb,
            pl.col("geo_type") == "state",
            pl.col("reference_date") >= first_training_date,
        )
        .select(
            [
                pl.col("reference_date").alias("date"),
                pl.col("geo_value").cast(pl.Utf8),
                pl.col("disease").cast(pl.Utf8),
                pl.col("value").alias("ed_visits"),
            ]
        )
        .with_columns(
            disease=pl.col("disease").cast(pl.Utf8).replace(_inverse_disease_map),
        )
        .sort(["date", "disease"])
        .collect(engine="streaming")
    )


def aggregate_facility_level_nssp_to_loc(
    facility_level_nssp_data: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    first_training_date: str,
    loc_pop_df: pl.DataFrame,
) -> pl.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if facility_level_nssp_data is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "geo_value": pl.Utf8,
                "disease": pl.Utf8,
                "ed_visits": pl.Float64,
            }
        )

    disease_key = _disease_map.get(disease, disease)

    if loc_abb == "US":
        logger.info("Aggregating facility-level data to national")
        locations_to_aggregate = (
            loc_pop_df.filter(pl.col("abb") != "US").get_column("abb").unique()
        )
        facility_level_nssp_data = aggregate_to_national(
            facility_level_nssp_data,
            locations_to_aggregate,
            first_training_date,
            national_geo_value="US",
        )

    return (
        facility_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == loc_abb,
            pl.col("reference_date") >= first_training_date,
        )
        .group_by(["reference_date", "disease"])
        .agg(pl.col("value").sum().alias("ed_visits"))
        .with_columns(
            disease=pl.col("disease").cast(pl.Utf8).replace(_inverse_disease_map),
            geo_value=pl.lit(loc_abb).cast(pl.Utf8),
        )
        .rename({"reference_date": "date"})
        .sort(["date", "disease"])
        .select(["date", "geo_value", "disease", "ed_visits"])
        .collect()
    )


def get_loc_pop_df():
    return forecasttools.location_table.select(
        pl.col("short_name").alias("abb"),
        pl.col("long_name").alias("name"),
        pl.col("population"),
    )


def _validate_and_extract(
    df: pl.DataFrame,
    parameter_name: str,
) -> list:
    df = df.filter(pl.col("parameter") == parameter_name).collect()
    if df.height != 1:
        error_msg = f"Expected exactly one {parameter_name} parameter row, but found {df.height}"
        logging.error(error_msg)
        if df.height > 0:
            logging.error(f"Found rows: {df}")
        raise ValueError(error_msg)
    return df.item(0, "value").to_list()


def get_pmfs(
    param_estimates: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    as_of: date = None,
    reference_date: date = None,
    right_truncation_required: bool = True,
) -> dict[str, list]:
    """
    Filter and extract probability mass functions (PMFs) from
    parameter estimates LazyFrame based on location, disease
    and date filters.

    This function queries a LazyFrame containing epidemiological
    parameters and returns a dictionary of three PMFs:
    delay, generation interval, and right truncation.

    Parameters
    ----------
    param_estimates: pl.LazyFrame
        A LazyFrame containing parameter data with columns
        including 'disease', 'parameter', 'value', 'geo_value',
        'start_date', 'end_date', and 'reference_date'.

    loc_abb : str
        Location abbreviation (geo_value) to filter
        right truncation parameters.

    disease : str
        Name of the disease.

    as_of : date, optional
        Date for which parameters must be valid
        (start_date <= as_of <= end_date). Defaults
        to the most recent estimates.

    reference_date : date, optional
        The reference date for right truncation estimates.
        Defaults to as_of value. Selects the most recent estimate
        with reference_date <= this value.

    right_truncation_required : bool, optional
        If False, allows extraction of other pmfs if
        right_truncation estimate is missing

    Returns
    -------
    dict[str, list]
        A dictionary containing three PMF arrays:
        - 'generation_interval_pmf': Generation interval distribution
        - 'delay_pmf': Delay distribution
        - 'right_truncation_pmf': Right truncation distribution

    Raises
    ------
    ValueError
        If exactly one row is not found for any of the required parameters.

    Notes
    -----
    The function applies specific filtering logic for each parameter type:
    - For delay and generation_interval: filters by disease,
      parameter name, and validity date range.
    - For right_truncation: additionally filters by location.
    """
    min_as_of = dt.date(1000, 1, 1)
    max_as_of = dt.date(3000, 1, 1)
    as_of = as_of or max_as_of
    reference_date = reference_date or as_of

    filtered_estimates = (
        param_estimates.with_columns(
            pl.col("start_date").fill_null(min_as_of),
            pl.col("end_date").fill_null(max_as_of),
        )
        .filter(pl.col("disease") == disease)
        .filter(
            pl.col("start_date") <= as_of,
            pl.col("end_date") >= as_of,
        )
    )

    generation_interval_pmf = _validate_and_extract(
        filtered_estimates, "generation_interval"
    )

    delay_pmf = _validate_and_extract(filtered_estimates, "delay")

    # ensure 0 first entry; we do not model the possibility
    # of a zero infection-to-recorded-admission delay in Pyrenew-HEW
    delay_pmf[0] = 0.0
    delay_pmf = jnp.array(delay_pmf)
    delay_pmf = delay_pmf / delay_pmf.sum()
    delay_pmf = delay_pmf.tolist()

    right_truncation_df = filtered_estimates.filter(
        pl.col("geo_value") == loc_abb
    ).filter(pl.col("reference_date") == pl.col("reference_date").max())

    if right_truncation_df.collect().height == 0 and not right_truncation_required:
        right_truncation_pmf = [1]
    else:
        right_truncation_pmf = _validate_and_extract(
            right_truncation_df, "right_truncation"
        )

    return {
        "generation_interval_pmf": generation_interval_pmf,
        "delay_pmf": delay_pmf,
        "right_truncation_pmf": right_truncation_pmf,
    }


def process_and_save_loc_data(
    loc_abb: str,
    disease: str,
    report_date: date,
    first_training_date: date,
    last_training_date: date,
    save_dir: Path,
    logger: Logger | None = None,
    facility_level_nssp_data: pl.LazyFrame | None = None,
    loc_level_nssp_data: pl.LazyFrame | None = None,
    loc_level_nwss_data: pl.DataFrame | None = None,
    credentials_dict: dict | None = None,
    nhsn_data_path: Path | str | None = None,
) -> None:
    """
    Process and save surveillance data for a specific location and disease for model fitting.
    
    This is the main data processing pipeline that combines NSSP emergency department visits,
    NHSN hospital admissions, and optionally NWSS wastewater data into a standardized format
    suitable for epidemiological modeling. Saves the processed data as JSON for model input.

    Parameters
    ----------
    loc_abb : str
        Location abbreviation (e.g., "CA", "TX", "US" for national level)
    disease : str
        Disease name, must be one of: "COVID-19", "Influenza", "RSV"
    report_date : date
        Reference date for the analysis (typically the current date)
    first_training_date : date
        Start date for the training data period
    last_training_date : date
        End date for the training data period
    save_dir : Path
        Directory where processed data files will be saved
    logger : Logger, optional
        Logger instance for tracking processing steps. If None, creates a new logger
    facility_level_nssp_data : pl.LazyFrame, optional
        Facility-level NSSP emergency department visit data
    loc_level_nssp_data : pl.LazyFrame, optional  
        State/location-level NSSP emergency department visit data
    loc_level_nwss_data : pl.LazyFrame, optional
        Location-level NWSS wastewater surveillance data
    credentials_dict : dict, optional
        API credentials for accessing NHSN data. Contains keys:
        - "nhsn_api_key_id": CDC API key ID
        - "nhsn_api_key_secret": CDC API key secret
    nhsn_data_path : Path | str, optional
        Path to local NHSN data file instead of API download

    Raises
    ------
    ValueError
        If neither facility_level_nssp_data nor loc_level_nssp_data is provided

    Notes
    -----
    - Creates save_dir if it doesn't exist
    - Processes each data stream separately then combines them
    - Handles data aggregation from facility to location level when needed
    - Saves final processed data as JSON file for model consumption
    - Uses population data for proper weighting and normalization
    - Filters data to specified training period for model fitting
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(save_dir, exist_ok=True)

    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            "Must provide at least one of facility-level and state-levelNSSP data"
        )

    loc_pop_df = get_loc_pop_df()

    loc_pop = loc_pop_df.filter(pl.col("abb") == loc_abb).item(0, "population")

    right_truncation_offset = (report_date - last_training_date).days - 1
    # First entry of source right truncation PMFs corresponds to reports
    # for ref date = report_date - 1 as of report_date

    aggregated_facility_data = aggregate_facility_level_nssp_to_loc(
        facility_level_nssp_data=facility_level_nssp_data,
        loc_abb=loc_abb,
        disease=disease,
        first_training_date=first_training_date,
        loc_pop_df=loc_pop_df,
    )

    loc_level_data = process_loc_level_data(
        loc_level_nssp_data=loc_level_nssp_data,
        loc_abb=loc_abb,
        disease=disease,
        first_training_date=first_training_date,
        loc_pop_df=loc_pop_df,
    )

    if aggregated_facility_data.height > 0:
        first_facility_level_data_date = aggregated_facility_data.get_column(
            "date"
        ).min()
        loc_level_data = loc_level_data.filter(
            pl.col("date") < first_facility_level_data_date
        )

    nssp_training_data = (
        pl.concat([loc_level_data, aggregated_facility_data])
        .filter(pl.col("date") <= last_training_date)
        .with_columns(pl.lit("train").alias("data_type"))
        .pivot(
            on="disease",
            values="ed_visits",
        )
        .rename({disease: "observed_ed_visits", "Total": "other_ed_visits"})
        .sort("date")
    )

    nhsn_training_data = (
        get_nhsn(
            start_date=first_training_date,
            end_date=last_training_date,
            disease=disease,
            loc_abb=loc_abb,
            credentials_dict=credentials_dict,
            local_data_file=nhsn_data_path,
        )
        .filter(
            (pl.col("weekendingdate") <= last_training_date)
            & (pl.col("weekendingdate") >= first_training_date)
        )  # in testing mode, this isn't guaranteed
        .with_columns(pl.lit("train").alias("data_type"))
    )

    nhsn_step_size = 7

    nwss_training_data = (
        loc_level_nwss_data.to_dict(as_series=False)
        if loc_level_nwss_data is not None
        else None
    )

    data_for_model_fit = {
        "loc_pop": loc_pop,
        "right_truncation_offset": right_truncation_offset,
        "nwss_training_data": nwss_training_data,
        "nssp_training_data": nssp_training_data.to_dict(as_series=False),
        "nhsn_training_data": nhsn_training_data.to_dict(as_series=False),
        "nhsn_step_size": nhsn_step_size,
        "nssp_step_size": 1,
        "nwss_step_size": 1,
    }

    with open(Path(save_dir, "data_for_model_fit.json"), "w") as json_file:
        json.dump(data_for_model_fit, json_file, default=str)

    combined_training_dat = combine_surveillance_data(
        nssp_data=nssp_training_data,
        nhsn_data=nhsn_training_data,
        nwss_data=loc_level_nwss_data if loc_level_nwss_data is not None else None,
        disease=disease,
    )

    if logger is not None:
        logger.info(f"Saving {loc_abb} to {save_dir}")

    combined_training_dat.write_csv(
        Path(save_dir, "combined_training_data.tsv"), separator="\t"
    )
    return None


def process_and_save_loc_param(
    loc_abb,
    disease,
    loc_level_nwss_data,
    param_estimates,
    fit_ed_visits,
    save_dir,
) -> None:
    loc_pop_df = get_loc_pop_df()
    loc_pop = loc_pop_df.filter(pl.col("abb") == loc_abb).item(0, "population")

    if loc_level_nwss_data is None:
        pop_fraction = jnp.array([1])
    else:
        subpop_sizes = (
            loc_level_nwss_data.select(["site_index", "site", "site_pop"])
            .unique()
            .sort("site_pop", descending=True)
            .get_column("site_pop")
            .to_numpy()
        )
        if loc_pop > sum(subpop_sizes):
            pop_fraction = (
                jnp.concatenate(
                    (jnp.array([loc_pop - sum(subpop_sizes)]), subpop_sizes)
                )
                / loc_pop
            )
        else:
            pop_fraction = subpop_sizes / sum(subpop_sizes)

    pmfs = get_pmfs(
        param_estimates=param_estimates,
        loc_abb=loc_abb,
        disease=disease,
        right_truncation_required=fit_ed_visits,
    )

    inf_to_hosp_admit_lognormal_loc, inf_to_hosp_admit_lognormal_scale = approx_lognorm(
        jnp.array(pmfs["delay_pmf"])[1:],  # only fit the non-zero delays
        loc_guess=0,
        scale_guess=0.5,
    )

    model_params = {
        "population_size": loc_pop,
        "pop_fraction": pop_fraction.tolist(),
        "generation_interval_pmf": pmfs["generation_interval_pmf"],
        "right_truncation_pmf": pmfs["right_truncation_pmf"],
        "inf_to_hosp_admit_lognormal_loc": inf_to_hosp_admit_lognormal_loc,
        "inf_to_hosp_admit_lognormal_scale": inf_to_hosp_admit_lognormal_scale,
        "inf_to_hosp_admit_pmf": pmfs["delay_pmf"],
    }
    with open(Path(save_dir, "model_params.json"), "w") as json_file:
        json.dump(model_params, json_file, default=str)

    return None
