#!/usr/bin/env python3
"""
Script to create EpiAutoGP JSON input from NHSN hospital admissions data.

This script uses the get_nhsn function from prep_data to retrieve hospital admissions
data and creates a JSON file in the EpiAutoGPInput format required by the Julia EpiAutoGP model.
The NHSN data contains hospital admissions by disease, location, and epidemiological week.
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

from pipeline.prep_data import get_nhsn


def prepare_nhsn_time_series_data(df: pl.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Prepare NHSN time series data for EpiAutoGP format.
    
    Args:
        df: NHSN DataFrame with columns: weekendingdate, jurisdiction, hospital_admissions
        
    Returns:
        Tuple of (dates as ISO strings, values as floats)
    """
    # Sort by date to ensure chronological order
    df_sorted = df.sort('weekendingdate')
    
    # Convert dates to ISO format strings
    dates = df_sorted['weekendingdate'].dt.strftime('%Y-%m-%d').to_list()
    
    # Convert values to float, handling any nulls
    values = df_sorted['hospital_admissions'].cast(pl.Float64).fill_null(0.0).to_list()
    
    return dates, values


def create_nhsn_epiautogp_json(
    dates: List[str],
    values: List[float],
    pathogen: str = "COVID-19",
    location: str = "US",
    target: str = "nhsn_hospital_admissions",
    forecast_date: Optional[str] = None
) -> Dict:
    """
    Create the EpiAutoGP JSON structure for NHSN data.
    
    Args:
        dates: List of date strings in ISO format
        values: List of corresponding hospital admission values
        pathogen: Pathogen name
        location: Location identifier
        target: Target metric name
        forecast_date: Forecast date (defaults to today)
        
    Returns:
        Dictionary in EpiAutoGP format
    """
    if forecast_date is None:
        forecast_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create the JSON structure
    epiautogp_data = {
        "dates": dates,
        "reports": values,
        "pathogen": pathogen,
        "location": location,
        "target": target,
        "forecast_date": forecast_date,
        "nowcast_dates": [],  # Always empty as requested
        "nowcast_reports": []  # Empty as requested
    }
    
    return epiautogp_data


def convert_nhsn_to_epiautogp(
    start_date: str,
    end_date: str,
    disease: str = "COVID-19",
    loc_abb: str = "US",
    output_path: str = "nhsn_output.json",
    pathogen: Optional[str] = None,
    target: str = "nhsn_hospital_admissions",
    forecast_date: Optional[str] = None,
    temp_dir: Optional[Path] = None,
    credentials_dict: Optional[dict] = None,
    local_data_file: Optional[Path] = None
) -> Dict:
    """
    Main function to convert NHSN data to EpiAutoGP JSON format.
    
    Args:
        start_date: Start date for data retrieval (YYYY-MM-DD format)
        end_date: End date for data retrieval (YYYY-MM-DD format)
        disease: Disease name ("COVID-19", "Influenza", or "RSV")
        loc_abb: Location abbreviation (e.g., "US", "CA", "TX")
        output_path: Path where JSON will be saved
        pathogen: Pathogen name for JSON (defaults to disease value)
        target: Target metric name for JSON
        forecast_date: Forecast date for JSON
        temp_dir: Directory for temporary files
        credentials_dict: API credentials dictionary
        local_data_file: Path to local parquet file instead of API call
        
    Returns:
        The created EpiAutoGP data dictionary
    """
    if pathogen is None:
        pathogen = disease
    
    # Convert string dates to date objects
    start_date_obj = date.fromisoformat(start_date)
    end_date_obj = date.fromisoformat(end_date)
    
    print(f"Retrieving NHSN data for {disease} in {loc_abb}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Get NHSN data using prep_data function
    nhsn_df = get_nhsn(
        start_date=start_date_obj,
        end_date=end_date_obj,
        disease=disease,
        loc_abb=loc_abb,
        temp_dir=temp_dir,
        credentials_dict=credentials_dict,
        local_data_file=local_data_file
    )
    
    print(f"Retrieved NHSN data shape: {nhsn_df.shape}")
    print(f"Columns: {nhsn_df.columns}")
    
    if nhsn_df.height == 0:
        raise ValueError(f"No NHSN data found for the specified parameters")
    
    # Display data summary
    print(f"Date range in data: {nhsn_df['weekendingdate'].min()} to {nhsn_df['weekendingdate'].max()}")
    admissions_stats = nhsn_df['hospital_admissions'].describe()
    print(f"Hospital admissions statistics:")
    for stat in admissions_stats.iter_rows(named=True):
        print(f"  {stat['statistic']}: {stat['value']}")
    
    # Prepare time series data
    dates, values = prepare_nhsn_time_series_data(nhsn_df)
    
    print(f"Time series length: {len(dates)} observations")
    print(f"Value range: {min(values):.1f} to {max(values):.1f}")
    
    # Create EpiAutoGP JSON
    epiautogp_data = create_nhsn_epiautogp_json(
        dates=dates,
        values=values,
        pathogen=pathogen,
        location=loc_abb,
        target=target,
        forecast_date=forecast_date
    )
    
    # Save to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(epiautogp_data, f, indent=2)
    
    print(f"EpiAutoGP JSON saved to {output_path}")
    
    return epiautogp_data


def main():
    """Command line interface for NHSN to EpiAutoGP conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert NHSN hospital admissions data to EpiAutoGP JSON format"
    )
    
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for data retrieval (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date for data retrieval (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--disease",
        default="COVID-19",
        choices=["COVID-19", "Influenza", "RSV"],
        help="Disease name (default: COVID-19)"
    )
    
    parser.add_argument(
        "--loc-abb",
        default="US",
        help="Location abbreviation (e.g., US, CA, TX) (default: US)"
    )
    
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where EpiAutoGP JSON will be saved"
    )
    
    parser.add_argument(
        "--pathogen",
        help="Pathogen name for JSON (defaults to disease value)"
    )
    
    parser.add_argument(
        "--target",
        default="nhsn_hospital_admissions",
        help="Target metric name for JSON (default: nhsn_hospital_admissions)"
    )
    
    parser.add_argument(
        "--forecast-date",
        help="Forecast date for JSON (YYYY-MM-DD). Defaults to today"
    )
    
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary files"
    )
    
    parser.add_argument(
        "--local-data-file",
        help="Path to local parquet file instead of API call"
    )
    
    parser.add_argument(
        "--api-key-id",
        help="NHSN API key ID (can also use NHSN_API_KEY_ID environment variable)"
    )
    
    parser.add_argument(
        "--api-key-secret",
        help="NHSN API key secret (can also use NHSN_API_KEY_SECRET environment variable)"
    )
    
    args = parser.parse_args()
    
    # Prepare credentials dictionary if provided
    credentials_dict = None
    if args.api_key_id or args.api_key_secret:
        credentials_dict = {}
        if args.api_key_id:
            credentials_dict["nhsn_api_key_id"] = args.api_key_id
        if args.api_key_secret:
            credentials_dict["nhsn_api_key_secret"] = args.api_key_secret
    
    # Prepare temp_dir and local_data_file paths
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    local_data_file = Path(args.local_data_file) if args.local_data_file else None
    
    convert_nhsn_to_epiautogp(
        start_date=args.start_date,
        end_date=args.end_date,
        disease=args.disease,
        loc_abb=args.loc_abb,
        output_path=args.output_path,
        pathogen=args.pathogen,
        target=args.target,
        forecast_date=args.forecast_date,
        temp_dir=temp_dir,
        credentials_dict=credentials_dict,
        local_data_file=local_data_file
    )


if __name__ == "__main__":
    main()