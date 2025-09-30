import polars as pl


def clean_nwss_data(nwss_data):
    """
    Clean and preprocess NWSS (National Wastewater Surveillance System) data for modeling.
    
    This function filters NWSS data to wastewater treatment plants (WWTP) only, 
    standardizes units to copies/ml, handles values below the limit of detection (LOD),
    and removes duplicate measurements.

    Parameters
    ----------
    nwss_data : pl.DataFrame
        Raw/vintaged NWSS data containing wastewater surveillance measurements
        with columns including sample_location, sample_matrix, pcr_target_units,
        pcr_target, lab_id, wwtp_id, lod_sewage, etc.

    Returns
    -------
    pl.DataFrame
        Cleaned site-lab level dataset with standardized columns:
        - date: sample collection date
        - site: WWTP identifier  
        - lab: laboratory identifier
        - log_genome_copies_per_ml: log-transformed viral concentration
        - log_lod: log-transformed limit of detection
        - site_pop: population served by WWTP
        - location: jurisdiction (state/territory)
    
    Notes
    -----
    - Filters to WWTP samples only (excludes primary sludge)
    - Converts all concentration units to copies/ml
    - Replaces values below LOD with 0.5 * LOD
    - Uses time-averaged population for sites with varying population
    - Removes exact duplicate measurements for same site-lab-date combinations
    """
    nwss_subset = (
        nwss_data.filter(
            pl.col("sample_location") == "wwtp",
            pl.col("sample_matrix") != "primary sludge",
            pl.col("pcr_target_units") != "copies/g dry sludge",
            pl.col("pcr_target") == "sars-cov-2",
            pl.col("lab_id").is_not_null(),
            pl.col("wwtp_id").is_not_null(),
            pl.col("lod_sewage").is_not_null(),
        )
        .select(
            [
                "lab_id",
                "sample_collect_date",
                "wwtp_id",
                "pcr_target_avg_conc",
                "wwtp_jurisdiction",
                "population_served",
                "pcr_target_units",
                "lod_sewage",
                "quality_flag",
            ]
        )
        .with_columns(
            pcr_target_avg_conc=pl.when(
                pl.col("pcr_target_units") == "copies/l wastewater"
            )
            .then(pl.col("pcr_target_avg_conc") / 1000)
            .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
            .then((10 ** pl.col("pcr_target_avg_conc")) / 1000)
            .otherwise(None),
            lod_sewage=pl.when(pl.col("pcr_target_units") == "copies/l wastewater")
            .then(pl.col("lod_sewage") / 1000)
            .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
            .then((10 ** pl.col("lod_sewage")) / 1000)
            .otherwise(None),
        )
        .filter(
            (
                ~pl.col("quality_flag").is_in(
                    [
                        "yes",
                        "y",
                        "result is not quantifiable",
                        "temperature not assessed upon arrival at the laboratory",
                        "> max temp and/or hold time",
                    ]
                )
            )
            | (pl.col("quality_flag").is_null()),
            pl.col("pcr_target_avg_conc") >= 0,
        )
    ).drop(["quality_flag", "pcr_target_units"])

    # Remove if any exact duplicates of pcr_target_avg_conc
    # values present for each combination of wwtp_id, lab_id,
    # and sample_collect_date
    nwss_subset_clean = nwss_subset.unique(
        subset=[
            "sample_collect_date",
            "wwtp_id",
            "lab_id",
            "pcr_target_avg_conc",
        ]
    )

    # If pcr_target_avg_conc is less than LOD, replace with 0.5*LOD
    nwss_subset_clean = nwss_subset_clean.with_columns(
        pcr_target_avg_conc=pl.when(
            pl.col("pcr_target_avg_conc") < pl.col("lod_sewage")
        )
        .then(0.5 * pl.col("lod_sewage"))
        .otherwise(pl.col("pcr_target_avg_conc"))
    )

    # replaces time-varying population if present in the NWSS dataset.
    # Model does not allow time varying population
    nwss_subset_clean_pop = (
        nwss_subset_clean.group_by("wwtp_id")
        .agg(
            [
                pl.col("population_served")
                .mean()
                .round()
                .cast(pl.Int64)
                .alias("population_served")
            ]
        )
        .join(nwss_subset_clean, on=["wwtp_id"], how="left")
        .select(
            [
                "sample_collect_date",
                "wwtp_id",
                "lab_id",
                "pcr_target_avg_conc",
                "wwtp_jurisdiction",
                "lod_sewage",
                "population_served",
            ]
        )
        .unique(
            [
                "wwtp_id",
                "lab_id",
                "sample_collect_date",
                "pcr_target_avg_conc",
            ]
        )
    )

    ww_data = (
        nwss_subset_clean_pop.rename(
            {
                "sample_collect_date": "date",
                "population_served": "site_pop",
                "wwtp_jurisdiction": "location",
                "wwtp_id": "site",
                "lab_id": "lab",
            }
        )
        .with_columns(
            [
                pl.col("pcr_target_avg_conc").log().alias("log_genome_copies_per_ml"),
                pl.col("lod_sewage").log().alias("log_lod"),
                pl.col("location").str.to_uppercase().alias("location"),
                pl.col("site").cast(pl.String).alias("site"),
                pl.col("lab").cast(pl.String).alias("lab"),
            ]
        )
        .select(
            [
                "date",
                "site",
                "lab",
                "log_genome_copies_per_ml",
                "log_lod",
                "site_pop",
                "location",
            ]
        )
    )
    return ww_data


def check_missing_values(df: pl.DataFrame, columns: list[str]):
    """
    Validate that specified columns in a DataFrame contain no missing values.
    
    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to check for missing values
    columns : list[str]
        List of column names to validate for completeness
        
    Raises
    ------
    ValueError
        If any of the specified columns contain null/missing values,
        with details about which columns have missing data
    """
    missing_cols = [col for col in columns if df[col].has_nulls()]
    if missing_cols:
        raise ValueError(f"Missing values in column(s): {missing_cols}")


def validate_ww_conc_data(
    ww_data: pl.DataFrame,
    conc_col_name: str = "log_genome_copies_per_ml",
    lod_col_name: str = "log_lod",
    date_col_name: str = "date",
    wwtp_col_name: str = "site",
    wwtp_pop_name: str = "site_pop",
    lab_col_name: str = "lab",
):
    """
    Comprehensive validation of wastewater concentration data structure and content.
    
    Validates that the DataFrame contains all required columns with appropriate data types,
    no missing values, and logical constraints (e.g., positive populations, consistent
    site populations over time).

    Parameters
    ----------
    ww_data : pl.DataFrame
        Wastewater concentration data to validate
    conc_col_name : str, default "log_genome_copies_per_ml"
        Column name for log-transformed viral concentration measurements
    lod_col_name : str, default "log_lod"
        Column name for log-transformed limit of detection values
    date_col_name : str, default "date"
        Column name for sample collection dates
    wwtp_col_name : str, default "site"
        Column name for wastewater treatment plant identifiers
    wwtp_pop_name : str, default "site_pop"
        Column name for population served by each WWTP
    lab_col_name : str, default "lab"
        Column name for laboratory identifiers

    Raises
    ------
    ValueError
        If DataFrame is empty, required columns are missing, columns have
        incorrect data types, missing values are present, populations are
        negative, or sites have varying population sizes over time
        
    Returns
    -------
    None
        Function performs validation only; returns None if all checks pass
    """
    if ww_data.is_empty():
        raise ValueError("Input DataFrame 'ww_data' is empty.")

    required_cols = [
        conc_col_name,
        lod_col_name,
        date_col_name,
        wwtp_col_name,
        wwtp_pop_name,
        lab_col_name,
    ]

    assert all(col in ww_data.columns for col in required_cols), (
        "One or more required column(s) missing"
    )

    check_missing_values(
        ww_data,
        required_cols,
    )

    assert ww_data[conc_col_name].dtype.is_float()
    assert ww_data[lod_col_name].dtype.is_float()
    assert ww_data[date_col_name].dtype == pl.Date
    assert ww_data[wwtp_pop_name].dtype.is_integer()
    assert ww_data[wwtp_col_name].dtype == pl.String()
    assert ww_data[lab_col_name].dtype == pl.String()

    if (ww_data[wwtp_pop_name] < 0).any():
        raise ValueError("Site populations have negative values.")

    if (
        not ww_data.group_by(wwtp_col_name)
        .n_unique()
        .get_column(wwtp_pop_name)
        .eq(1)
        .all()
    ):
        raise ValueError("The data contains sites with varying population sizes.")

    return None


def preprocess_ww_data(
    ww_data,
    conc_col_name: str = "log_genome_copies_per_ml",
    lod_col_name: str = "log_lod",
    date_col_name: str = "date",
    wwtp_col_name: str = "site",
    wwtp_pop_name: str = "site_pop",
    lab_col_name: str = "lab",
):
    """
    Preprocess wastewater data by creating indices and flagging below-detection measurements.
    
    Creates standardized indices for wastewater treatment plants and lab-site combinations,
    ordered by population size (largest first). Adds flags for measurements below the
    limit of detection and creates human-readable site-lab identifiers.

    Parameters
    ----------
    ww_data : pl.DataFrame
        Validated wastewater concentration data
    conc_col_name : str, default "log_genome_copies_per_ml"
        Column name for log-transformed viral concentration measurements
    lod_col_name : str, default "log_lod"
        Column name for log-transformed limit of detection values
    date_col_name : str, default "date"
        Column name for sample collection dates
    wwtp_col_name : str, default "site"
        Column name for wastewater treatment plant identifiers
    wwtp_pop_name : str, default "site_pop"
        Column name for population served by each WWTP
    lab_col_name : str, default "lab"
        Column name for laboratory identifiers

    Returns
    -------
    pl.DataFrame
        Preprocessed wastewater data with additional columns:
        - site_index: integer index for each unique WWTP (ordered by population)
        - lab_site_index: integer index for each unique lab-site combination
        - lab_site_name: human-readable "Site: X, Lab: Y" identifier
        - below_lod: boolean flag for measurements below limit of detection
        
    Notes
    -----
    - Sites and lab-site combinations are indexed in descending order of population
    - This ordering prioritizes larger population centers in modeling
    - Validates input data using validate_ww_conc_data() before processing
    """
    validate_ww_conc_data(
        ww_data,
        conc_col_name=conc_col_name,
        lod_col_name=lod_col_name,
        date_col_name=date_col_name,
    )
    lab_site_df = (
        ww_data.select([lab_col_name, wwtp_col_name, wwtp_pop_name])
        .unique()
        .sort(by=wwtp_pop_name, descending=True)
        .with_row_index("lab_site_index")
    )
    site_df = (
        ww_data.select([wwtp_col_name, wwtp_pop_name])
        .unique()
        .sort(by=wwtp_pop_name, descending=True)
        .with_row_index("site_index")
    )
    ww_preprocessed = (
        ww_data.sort(by=wwtp_pop_name, descending=True)
        .join(
            lab_site_df,
            on=[lab_col_name, wwtp_col_name, wwtp_pop_name],
            how="left",
        )
        .join(site_df, on=[wwtp_col_name, wwtp_pop_name], how="left")
        .rename(
            {
                lod_col_name: "log_lod",
                conc_col_name: "log_genome_copies_per_ml",
            }
        )
        .with_columns(
            lab_site_name=(
                "Site: " + pl.col(wwtp_col_name) + ", Lab: " + pl.col(lab_col_name)
            ),
            below_lod=(pl.col("log_genome_copies_per_ml") <= pl.col("log_lod")),
        )
        .select(
            [
                "date",
                "site",
                "lab",
                "location",
                "site_pop",
                "site_index",
                "lab_site_name",
                "lab_site_index",
                "log_genome_copies_per_ml",
                "log_lod",
                "below_lod",
            ]
        )
    )
    return ww_preprocessed
