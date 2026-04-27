import pandas as pd, pathlib, argparse

COLS = [
    # Identity / location
    "grid_id", "analysis_neighborhood", "block_address",
    "supervisor_district", "lat", "lon", "submission_year",
    # Rent & unit details
    "monthly_rent_clean", "bedrooms_clean", "bathrooms_clean",
    "sqft_avg", "unit_count_clean",
    "rent_per_sqft", "rent_per_resident",
    "sqft_per_resident", "bathrooms_per_resident",
    # Utilities included
    "base_rent_includes_water_clean", "base_rent_includes_natural_gas_clean",
    "base_rent_includes_electricity_clean", "base_rent_includes_refuse_recycling_clean",
    "utilities_included_count",
    # Tenure & property
    "occupancy_duration_years", "property_age", "year_property_built",
    "likely_rent_controlled",
    # 311 summary
    "total_311_requests", "avg_resolution_days", "median_resolution_days",
    "num_unique_services", "top_service",
    "pct_parking_enforcement",
    "pct_mta_parking_traffic_signs_high_priority",
    "pct_mta_parking_traffic_signs_normal_priority",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="docs/outputs/merged_rent_311.csv")
    ap.add_argument("--output", default="docs/outputs/rent_listings_slim.csv")
    args = ap.parse_args()

    print(f"Reading {args.input} (this takes ~30s)...")
    df = pd.read_csv(args.input, usecols=COLS, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    df.to_csv(args.output, index=False)

    size_mb = pathlib.Path(args.output).stat().st_size / 1_000_000
    print(f"  Saved {len(df):,} rows → {args.output} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
