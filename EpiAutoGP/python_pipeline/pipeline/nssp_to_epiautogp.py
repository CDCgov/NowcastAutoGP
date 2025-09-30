#!/usr/bin/env python3
"""
Script to create EpiAutoGP JSON input from NSSP ETL data.

This script processes the NSSP comprehensive parquet data and creates
a JSON file in the EpiAutoGPInput format required by the Julia EpiAutoGP model.
The NSSP data contains ED visits by disease, location, and time.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_nssp_data(parquet_path: str) -> pd.DataFrame:
    """
    Load NSSP comprehensive data from parquet file.
    
    Args:
        parquet_path: Path to the NSSP comprehensive parquet file
        
    Returns:
        DataFrame with NSSP data
    """
    df = pd.read_parquet(parquet_path)
    
    # Convert date columns to datetime
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['report_date'] = pd.to_datetime(df['report_date'])
    
    return df


def filter_nssp_data(
    df: pd.DataFrame,
    disease: str = "COVID-19/Omicron",
    metric: str = "count_ed_visits",
    geo_type: str = "state",
    geo_value: Optional[str] = None,
    report_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter NSSP data by specified criteria.
    
    Args:
        df: NSSP DataFrame
        disease: Disease to filter for (default: "COVID-19/Omicron")
        metric: Metric to filter for (default: "count_ed_visits")
        geo_type: Geographic type to filter for (default: "state")
        geo_value: Specific geographic value (e.g., "US", "CA"). If None, aggregates all
        report_date: Specific report date to filter for. If None, uses latest available
        
    Returns:
        Filtered DataFrame
    """
    # Filter by disease and metric
    filtered_df = df[
        (df['disease'] == disease) & 
        (df['metric'] == metric) &
        (df['geo_type'] == geo_type)
    ].copy()
    
    if filtered_df.empty:
        raise ValueError(f"No data found for disease='{disease}', metric='{metric}', geo_type='{geo_type}'")
    
    # Filter by specific geography if provided
    if geo_value is not None:
        filtered_df = filtered_df[filtered_df['geo_value'] == geo_value]
        if filtered_df.empty:
            raise ValueError(f"No data found for geo_value='{geo_value}'")
    
    # Filter by report date if provided, otherwise use latest
    if report_date is not None:
        report_date_dt = pd.to_datetime(report_date)
        filtered_df = filtered_df[filtered_df['report_date'] == report_date_dt]
        if filtered_df.empty:
            raise ValueError(f"No data found for report_date='{report_date}'")
    else:
        # Use the latest report date
        latest_report_date = filtered_df['report_date'].max()
        filtered_df = filtered_df[filtered_df['report_date'] == latest_report_date]
        print(f"Using latest report date: {latest_report_date.strftime('%Y-%m-%d')}")
    
    return filtered_df


def aggregate_geographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data across all geographic units for a given time series.
    
    Args:
        df: Filtered NSSP DataFrame
        
    Returns:
        DataFrame aggregated by reference_date with total values
    """
    # Group by reference_date and sum values across all geographic units
    aggregated_df = df.groupby('reference_date').agg({
        'value': 'sum',
        'report_date': 'first',  # Should be the same for all rows
        'disease': 'first',
        'metric': 'first',
        'geo_type': 'first'
    }).reset_index()
    
    # Sort by reference_date
    aggregated_df = aggregated_df.sort_values('reference_date')
    
    return aggregated_df


def prepare_time_series_data(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Prepare time series data for EpiAutoGP format.
    
    Args:
        df: Processed NSSP DataFrame (should be aggregated if needed)
        
    Returns:
        Tuple of (dates as ISO strings, values as floats)
    """
    # Sort by reference_date to ensure chronological order
    df_sorted = df.sort_values('reference_date')
    
    # Convert dates to ISO format strings
    dates = df_sorted['reference_date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Convert values to float
    values = df_sorted['value'].astype(float).tolist()
    
    return dates, values


def determine_nowcast_dates(dates: List[str], lookback_days: int = 14) -> List[str]:
    """
    Determine which dates should be considered for nowcasting.
    
    Typically, the most recent 1-2 weeks of data are subject to reporting delays
    and would benefit from nowcasting.
    
    Args:
        dates: List of date strings in ISO format
        lookback_days: Number of days to look back for nowcasting (default: 14)
        
    Returns:
        List of dates that should be nowcast
    """
    if not dates:
        return []
    
    # Get the latest date
    latest_date = datetime.fromisoformat(dates[-1])
    
    # Calculate cutoff date
    cutoff_date = latest_date - timedelta(days=lookback_days)
    
    # Find dates that are more recent than cutoff
    nowcast_dates = []
    for date_str in dates:
        date_obj = datetime.fromisoformat(date_str)
        if date_obj > cutoff_date:
            nowcast_dates.append(date_str)
    
    return nowcast_dates


def create_epiautogp_json(
    dates: List[str],
    values: List[float],
    nowcast_dates: List[str],
    pathogen: str = "COVID-19",
    location: str = "US",
    target: str = "nssp_ed_visits",
    forecast_date: Optional[str] = None
) -> Dict:
    """
    Create the EpiAutoGP JSON structure.
    
    Args:
        dates: List of date strings in ISO format
        values: List of corresponding values
        nowcast_dates: List of dates to be nowcast (ignored - always returns empty array)
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
        "nowcast_reports": []  # Empty for now as requested
    }
    
    return epiautogp_data


def convert_nssp_to_epiautogp(
    parquet_path: str,
    output_path: str,
    disease: str = "COVID-19/Omicron",
    metric: str = "count_ed_visits",
    geo_type: str = "state",
    geo_value: Optional[str] = None,
    report_date: Optional[str] = None,
    pathogen: str = "COVID-19",
    location: str = "US",
    target: str = "nssp_ed_visits",
    forecast_date: Optional[str] = None,
    nowcast_lookback_days: int = 14
) -> Dict:
    """
    Main function to convert NSSP data to EpiAutoGP JSON format.
    
    Args:
        parquet_path: Path to NSSP comprehensive parquet file
        output_path: Path where JSON will be saved
        disease: Disease to filter for
        metric: Metric to filter for
        geo_type: Geographic type to filter for
        geo_value: Specific geographic value (None = aggregate all)
        report_date: Specific report date (None = use latest)
        pathogen: Pathogen name for JSON
        location: Location identifier for JSON
        target: Target metric name for JSON
        forecast_date: Forecast date for JSON
        nowcast_lookback_days: Days to look back for nowcasting
        
    Returns:
        The created EpiAutoGP data dictionary
    """
    print(f"Loading NSSP data from {parquet_path}")
    df = load_nssp_data(parquet_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Available diseases: {sorted(df['disease'].unique())}")
    print(f"Available metrics: {sorted(df['metric'].unique())}")
    print(f"Available geo_types: {sorted(df['geo_type'].unique())}")
    
    # Filter the data
    print(f"Filtering for disease='{disease}', metric='{metric}', geo_type='{geo_type}'")
    if geo_value:
        print(f"Filtering for geo_value='{geo_value}'")
    
    filtered_df = filter_nssp_data(
        df, disease=disease, metric=metric, geo_type=geo_type, 
        geo_value=geo_value, report_date=report_date
    )
    
    print(f"Filtered data shape: {filtered_df.shape}")
    
    # Aggregate across geographies if needed (when geo_value is None)
    if geo_value is None:
        print("Aggregating data across all geographic units")
        processed_df = aggregate_geographic_data(filtered_df)
    else:
        processed_df = filtered_df
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Date range: {processed_df['reference_date'].min()} to {processed_df['reference_date'].max()}")
    
    # Prepare time series data
    dates, values = prepare_time_series_data(processed_df)
    
    print(f"Time series length: {len(dates)} observations")
    print(f"Value range: {min(values):.1f} to {max(values):.1f}")
    
    # Determine nowcast dates (not used since we return empty array)
    nowcast_dates = []  # Always empty as requested
    print(f"Nowcast dates: [] (empty as requested)")
    
    # Create EpiAutoGP JSON
    epiautogp_data = create_epiautogp_json(
        dates=dates,
        values=values,
        nowcast_dates=nowcast_dates,
        pathogen=pathogen,
        location=location if geo_value else location,  # Use provided location or "US" for aggregated
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
    """Command line interface for NSSP to EpiAutoGP conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert NSSP ETL data to EpiAutoGP JSON format"
    )
    
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to NSSP comprehensive parquet file"
    )
    
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where EpiAutoGP JSON will be saved"
    )
    
    parser.add_argument(
        "--disease",
        default="COVID-19/Omicron",
        help="Disease to filter for (default: COVID-19/Omicron)"
    )
    
    parser.add_argument(
        "--metric",
        default="count_ed_visits",
        help="Metric to filter for (default: count_ed_visits)"
    )
    
    parser.add_argument(
        "--geo-type",
        default="state",
        help="Geographic type to filter for (default: state)"
    )
    
    parser.add_argument(
        "--geo-value",
        help="Specific geographic value (e.g., CA, US). If not provided, aggregates all"
    )
    
    parser.add_argument(
        "--report-date",
        help="Specific report date (YYYY-MM-DD). If not provided, uses latest available"
    )
    
    parser.add_argument(
        "--pathogen",
        default="COVID-19",
        help="Pathogen name for JSON (default: COVID-19)"
    )
    
    parser.add_argument(
        "--location",
        default="US",
        help="Location identifier for JSON (default: US)"
    )
    
    parser.add_argument(
        "--target",
        default="nssp_ed_visits",
        help="Target metric name for JSON (default: nssp_ed_visits)"
    )
    
    parser.add_argument(
        "--forecast-date",
        help="Forecast date for JSON (YYYY-MM-DD). Defaults to today"
    )
    
    parser.add_argument(
        "--nowcast-lookback-days",
        type=int,
        default=14,
        help="Days to look back for nowcasting (default: 14)"
    )
    
    args = parser.parse_args()
    
    convert_nssp_to_epiautogp(
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        disease=args.disease,
        metric=args.metric,
        geo_type=args.geo_type,
        geo_value=args.geo_value,
        report_date=args.report_date,
        pathogen=args.pathogen,
        location=args.location,
        target=args.target,
        forecast_date=args.forecast_date,
        nowcast_lookback_days=args.nowcast_lookback_days
    )


if __name__ == "__main__":
    main()