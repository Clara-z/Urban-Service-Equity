from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

_GDRIVE_CSV = "/content/drive/My Drive/243 Group 2/Module 2/data/merged_rent_311.csv"
_LOCAL_CSV = "/Users/kylechang/Documents/Claude Projects/Urban-Service-Equity/merged_rent_311.csv"

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass


# -----------------------------
# Defaults (mirrors DB_MVP.ipynb)
# -----------------------------

def _resolve_default_csv() -> str:
    if env := os.environ.get("MERGED_RENT_311_CSV"):
        return env
    if os.path.isfile(_GDRIVE_CSV):
        return _GDRIVE_CSV
    return _LOCAL_CSV

DEFAULT_INPUT_CSV = _resolve_default_csv()

CLUSTER_FEATURES_DEFAULT: List[str] = [
    "median_rent",  # Economic baseline
    "median_resolution_days",  # Service efficiency
    "median_property_age",  # Infrastructure health
    "request_intensity",  # Systemic service load per unit
    "pct_311_external_request",  # Bureaucratic friction / referral rate
    "pct_street_and_sidewalk_cleaning",  # High-effort task share
]

SERVICE_COLS_DEFAULT: List[str] = [
    "total_311_requests",
    "avg_resolution_days",
    "median_resolution_days",
    "num_unique_services",
    "pct_street_and_sidewalk_cleaning",
    "pct_tree_maintenance",
    "pct_streetlights",
    "pct_rec_and_park_requests",
    "pct_graffiti",
    "pct_encampments",
    "pct_illegal_postings",
    "pct_abandoned_vehicle",
    "pct_noise_report",
]

NEED_COLS_DEFAULT: List[str] = [
    "unit_count_clean",
    "bedrooms_for_ratio",
    "sqft_avg",
    "sqft_per_resident",
    "bathrooms_per_resident",
    "property_age",
    "likely_rent_controlled",
    "monthly_rent_clean",
    "occupancy_duration_years",
]

CLUSTER_NAMES_DEFAULT: Dict[int, str] = {
    0: "Cluster A — High-Rent, Low Service Volume",
    1: "Cluster B — Dominant Residential",
    2: "Cluster C — High-Density Outlier",
    3: "Cluster D — Slow Resolution Hotspot",
}

# Heuristic archetypes (mirrors DB_MVP.ipynb Cell 10)
HEURISTICS_DEFAULT: Dict[int, dict] = {
    0: {
        "archetype": "High-Rent, Low Service Volume",
        "signal": "Newer stock + lower 311 demand — may mask under-reporting by renters",
        "needs": [
            {
                "rank": 1,
                "priority": "HIGH",
                "title": "Proactive Property Maintenance Outreach",
                "desc": "High rent + low service volume may mask deferred maintenance or tenant reluctance to report.",
                "actions": [
                    "Conduct proactive inspections in rent-controlled units",
                    "Deploy predictive maintenance alerts for aging infrastructure",
                    "Create landlord accountability portal for 311 requests",
                ],
            },
            {
                "rank": 2,
                "priority": "HIGH",
                "title": "Affordability & Displacement Prevention",
                "desc": "High rent with moderate service response. Residents may under-report fearing eviction.",
                "actions": [
                    "Expand multilingual tenant rights education and hotline",
                    "Partner with community orgs for confidential 311 submission",
                    "Track displacement-correlated service gaps longitudinally",
                ],
            },
            {
                "rank": 3,
                "priority": "MED",
                "title": "Service Volume Equity Audit",
                "desc": "Low 311 volume in high-rent areas may signal demand suppression, not satisfaction.",
                "actions": [
                    "Compare utilization vs. physical infrastructure condition",
                    "Investigate service equity for low-income renters in high-rent zones",
                    "Establish per-income-tier equity benchmarks within cluster",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Audit rent-controlled units for deferred maintenance backlog",
                "High rent + low volume is a suppression signal, not a success metric.",
            ),
            (
                "02",
                "Deploy multilingual 311 outreach in high-rent displacement corridors",
                "Non-English speakers in high-rent zones are least likely to self-report.",
            ),
            (
                "03",
                "Cross-reference 311 gaps with housing court filings",
                "Litigation spikes often precede 311 volume — early warning system.",
            ),
        ],
    },
    1: {
        "archetype": "Dominant Residential",
        "signal": "High volume, moderate affordability stress, elevated cleaning demand",
        "needs": [
            {
                "rank": 1,
                "priority": "HIGH",
                "title": "Street & Sidewalk Cleaning Scale-Up",
                "desc": "Highest cleaning request rate citywide — signals systemic under-staffing.",
                "actions": [
                    "Increase street sweeping frequency in high-density corridors",
                    "Install additional litter infrastructure at chokepoints",
                    "Pilot automated cleaning notification via 311 app",
                ],
            },
            {
                "rank": 2,
                "priority": "MED",
                "title": "Tree Canopy & Green Space Maintenance",
                "desc": "Elevated tree maintenance requests correlated with aging street trees.",
                "actions": [
                    "Prioritize tree inventory audit in blocks with highest 311 tree requests",
                    "Establish community stewardship programs for green corridors",
                    "Accelerate DPW tree trimming cycle from 10yr to 5yr target",
                ],
            },
            {
                "rank": 3,
                "priority": "HIGH",
                "title": "Housing Density Pressure Relief",
                "desc": "High occupancy ratio suggests units may be over-occupied relative to capacity.",
                "actions": [
                    "Expand secondary unit permitting in under-utilized parcels",
                    "Increase housing inspection capacity for overcrowding reports",
                    "Target density relief through inclusionary infill housing pipeline",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Surge street cleaning crews to top 20% density grid cells",
                "Cleaning requests are proportional to unmet daily maintenance burden.",
            ),
            (
                "02",
                "Fast-track tree trimming in corridors with 3+ open requests",
                "Deferred tree maintenance creates escalating liability and safety risk.",
            ),
            (
                "03",
                "Conduct housing overcrowding inspection sweep in flagged cells",
                "High density + long occupancy duration = silent overcrowding growth.",
            ),
        ],
    },
    2: {
        "archetype": "High-Density Outlier",
        "signal": "Extreme volume + severe crowding + minimal positive services — statistical anomaly",
        "needs": [
            {
                "rank": 1,
                "priority": "CRITICAL",
                "title": "Emergency High-Volume Service Infrastructure",
                "desc": "Exceptional 311 volume + critically low positive service delivery — system overwhelmed.",
                "actions": [
                    "Deploy dedicated service crew to this cell immediately",
                    "Triage 311 backlog by complaint age and recurrence frequency",
                    "Assess if this cell is a routing anomaly or genuine hotspot",
                ],
            },
            {
                "rank": 2,
                "priority": "CRITICAL",
                "title": "Crowding Investigation & Housing Audit",
                "desc": "Severe space crowding relative to building capacity. Likely undocumented multi-family.",
                "actions": [
                    "Immediate housing inspection referral",
                    "Cross-check building permits against occupancy data",
                    "Engage community health workers for vulnerable resident outreach",
                ],
            },
            {
                "rank": 3,
                "priority": "HIGH",
                "title": "Public Space Access Deficit",
                "desc": "Low positive service rate in high-density zone creates compounded livability pressure.",
                "actions": [
                    "Prioritize parklet or open space installation in adjacent blocks",
                    "Increase sidewalk maintenance inspection frequency",
                    "Establish community liaison for recurring public space issues",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Immediate multi-agency triage — 5sigma statistical outlier",
                "Volume + crowding + low services = highest composite risk in dataset.",
            ),
            (
                "02",
                "Deploy housing inspector and community health worker jointly",
                "Co-deployment reduces duplicate visits and builds resident trust.",
            ),
            (
                "03",
                "Escalate to supervisor district for emergency resource allocation",
                "Single-cell anomaly at this scale warrants political escalation.",
            ),
        ],
    },
    3: {
        "archetype": "Slow Resolution Hotspot",
        "signal": "Resolution time 5.5 sigma above city mean — institutional failure zone",
        "needs": [
            {
                "rank": 1,
                "priority": "CRITICAL",
                "title": "Emergency Resolution Time Intervention",
                "desc": "Resolution days are 5.5 standard deviations above city mean — systemic failure.",
                "actions": [
                    "Audit all open 311 tickets in this cluster older than 30 days",
                    "Assign dedicated case manager to chronic unresolved complaints",
                    "Auto-escalate any ticket open more than 14 days",
                ],
            },
            {
                "rank": 2,
                "priority": "HIGH",
                "title": "Encampment & Disorder Stabilization",
                "desc": "Elevated negative signals (encampments, graffiti, abandoned vehicles) compound slow resolution.",
                "actions": [
                    "Coordinate HSOC and DPW joint sweeps with rehousing referrals",
                    "Deploy graffiti abatement within 48h of report",
                    "Establish abandoned vehicle fast-track tow program",
                ],
            },
            {
                "rank": 3,
                "priority": "MED",
                "title": "Systemic Service Routing Reform",
                "desc": "Slow resolution often caused by inter-agency routing inefficiencies.",
                "actions": [
                    "Audit 311 routing logs for inter-agency handoff failures",
                    "Implement case ownership accountability by supervisor district",
                    "Create resolution SLA dashboard visible to department heads",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Audit all 311 tickets older than 30 days in Cluster D",
                "A 5.5sigma resolution delay is institutional failure, not backlog.",
            ),
            (
                "02",
                "Joint HSOC + DPW deployment for encampment clearance with rehousing",
                "Co-deployment with rehousing prevents cycling; clearance alone does not.",
            ),
            (
                "03",
                "Implement 14-day auto-escalation rule for all Cluster D tickets",
                "Without structural escalation triggers, chronic delays persist indefinitely.",
            ),
        ],
    },
}


@dataclass(frozen=True)
class Config:
    k: int = 4
    random_state: int = 42
    cluster_features: Tuple[str, ...] = tuple(CLUSTER_FEATURES_DEFAULT)
    service_cols: Tuple[str, ...] = tuple(SERVICE_COLS_DEFAULT)
    need_cols: Tuple[str, ...] = tuple(NEED_COLS_DEFAULT)
    cluster_names: Dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cluster_names is None:
            object.__setattr__(self, "cluster_names", dict(CLUSTER_NAMES_DEFAULT))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def load_and_aggregate_to_grid(csv_path: str) -> pd.DataFrame:
    """
    Unit-level CSV -> grid-level dataframe, matching DB_MVP.ipynb Cell 3.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    _require_columns(df, ["grid_id", "lat", "lon"], context="grid aggregation")

    pct_cols = [c for c in df.columns if c.startswith("pct_")]

    agg_dict: Dict[str, str] = {
        "lat": "mean",
        "lon": "mean",
        "monthly_rent_clean": "median",
        "avg_resolution_days": "median",
        "median_resolution_days": "median",
        "total_311_requests": "sum",
        "num_unique_services": "median",
        "property_age": "median",
        "likely_rent_controlled": "mean",
        "sqft_avg": "median",
        "sqft_per_resident": "median",
        "bathrooms_per_resident": "median",
        "unit_count_clean": "sum",
        "bedrooms_for_ratio": "median",
        "occupancy_duration_years": "median",
    }

    # Only aggregate columns that exist in the input schema
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    for c in pct_cols:
        agg_dict[c] = "mean"

    grid_df = df.groupby("grid_id").agg(agg_dict).reset_index()

    # Convenience aliases for clustering feature names
    if "monthly_rent_clean" in grid_df.columns:
        grid_df["median_rent"] = grid_df["monthly_rent_clean"]
    if "property_age" in grid_df.columns:
        grid_df["median_property_age"] = grid_df["property_age"]
    if "unit_count_clean" in grid_df.columns:
        grid_df["housing_density"] = grid_df["unit_count_clean"]

    # Derived feature: 311 requests per housing unit (systemic load indicator)
    if "total_311_requests" in grid_df.columns and "housing_density" in grid_df.columns:
        denom = grid_df["housing_density"].replace(0, 1)
        grid_df["request_intensity"] = grid_df["total_311_requests"] / denom

    # Parse row/col from grid_id (format: 'row_col')
    parts = grid_df["grid_id"].astype(str).str.split("_", expand=True)
    if parts.shape[1] >= 2:
        grid_df["grid_row"] = pd.to_numeric(parts[0], errors="coerce")
        grid_df["grid_col"] = pd.to_numeric(parts[1], errors="coerce")

    return grid_df


def run_kmeans(
    grid_df: pd.DataFrame, *, cfg: Config
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Match DB_MVP.ipynb Cell 4: standardize CLUSTER_FEATURES and KMeans(K=4).
    """
    _require_columns(grid_df, list(cfg.cluster_features), context="clustering")

    df_clust = grid_df.dropna(subset=list(cfg.cluster_features)).copy()
    scaler_c = StandardScaler()
    scaled_c = scaler_c.fit_transform(df_clust[list(cfg.cluster_features)])

    kmeans = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=10)
    df_clust["cluster"] = kmeans.fit_predict(scaled_c)

    return df_clust, kmeans, scaler_c


def equity_feature_engineering(df_clust: pd.DataFrame, *, cfg: Config) -> pd.DataFrame:
    """
    Match DB_MVP.ipynb Cell 6.
    """
    all_eq_cols = list(cfg.service_cols) + list(cfg.need_cols)
    _require_columns(df_clust, ["grid_id", "cluster", "lat", "lon"], context="equity scoring base")
    _require_columns(df_clust, all_eq_cols, context="equity scoring inputs")

    df_eq = df_clust[["grid_id", "cluster", "lat", "lon"] + all_eq_cols].dropna().copy()

    # Service Performance sub-indicators
    df_eq["S1"] = np.log1p(df_eq["total_311_requests"])
    df_eq["S2"] = (-0.5 * df_eq["avg_resolution_days"]) + (-0.5 * df_eq["median_resolution_days"])
    df_eq["S3"] = df_eq["num_unique_services"]
    df_eq["S4_pos"] = df_eq[
        [
            "pct_street_and_sidewalk_cleaning",
            "pct_tree_maintenance",
            "pct_streetlights",
            "pct_rec_and_park_requests",
        ]
    ].sum(axis=1)
    df_eq["S4_neg"] = df_eq[
        [
            "pct_encampments",
            "pct_graffiti",
            "pct_illegal_postings",
            "pct_abandoned_vehicle",
            "pct_noise_report",
        ]
    ].sum(axis=1)

    # Service Need sub-indicators
    df_eq["N1"] = (df_eq["unit_count_clean"] * df_eq["bedrooms_for_ratio"]) / df_eq["sqft_avg"].replace(
        0, np.nan
    )
    df_eq["N2"] = (1 / df_eq["sqft_per_resident"].replace(0, np.nan)) + (
        1 / df_eq["bathrooms_per_resident"].replace(0, np.nan)
    )
    df_eq["N3"] = df_eq["property_age"] + df_eq["likely_rent_controlled"]
    df_eq["N4"] = 1 / df_eq["monthly_rent_clean"].replace(0, np.nan)
    df_eq["N5"] = 1 / df_eq["occupancy_duration_years"].replace(0, np.nan)

    s_cols = ["S1", "S2", "S3", "S4_pos", "S4_neg"]
    n_cols = ["N1", "N2", "N3", "N4", "N5"]

    df_eq = df_eq[["grid_id", "cluster", "lat", "lon"] + s_cols + n_cols]
    df_eq = df_eq.replace([np.inf, -np.inf], np.nan).dropna()
    return df_eq


def compute_equity_scores(df_eq: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Match DB_MVP.ipynb Cell 7: PCA(n_components=1) per block, weighted sums,
    raw_equity ratio, log+clip+minmax -> 0-100 equity_score.
    """
    s_cols = ["S1", "S2", "S3", "S4_pos", "S4_neg"]
    n_cols = ["N1", "N2", "N3", "N4", "N5"]

    scaler_s = StandardScaler()
    scaler_n = StandardScaler()
    X_s = scaler_s.fit_transform(df_eq[s_cols])
    X_n = scaler_n.fit_transform(df_eq[n_cols])

    pca_s = PCA(n_components=1).fit(X_s)
    pca_n = PCA(n_components=1).fit(X_n)

    ws = pca_s.components_[0]
    wn = pca_n.components_[0]

    df_eq = df_eq.copy()
    df_eq["performance_score"] = (
        ws[0] * df_eq["S1"]
        + ws[1] * df_eq["S2"]
        + ws[2] * df_eq["S3"]
        + ws[3] * df_eq["S4_pos"]
        - ws[4] * df_eq["S4_neg"]
    )
    df_eq["need_score"] = (
        wn[0] * df_eq["N1"]
        + wn[1] * df_eq["N2"]
        + wn[2] * df_eq["N3"]
        + wn[3] * df_eq["N4"]
        + wn[4] * df_eq["N5"]
    )
    # Percentage Transformation here - Kai
    # try 3: log + clip + minmax -> 0-100
    df_eq["raw_equity"] = df_eq["performance_score"] / (df_eq["need_score"] + 1e-6)
    upper = np.percentile(df_eq["raw_equity"], 99)
    clipped = np.clip(df_eq["raw_equity"], df_eq["raw_equity"].min(), upper)
    df_eq["equity_score"] = ((clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)) * 100.0

    meta = {
        "pca_weights": {
            "service_performance": dict(zip(s_cols, ws.tolist())),
            "service_need": dict(zip(n_cols, wn.tolist())),
        }
    }
    return df_eq, meta


def zscore_feature_importance(df_eq: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Root-cause style signals: cluster mean z-scores vs global mean/std.
    Returns a z-score matrix (cluster x feature) and top-3 per cluster.
    """
    features = ["S1", "S2", "S3", "S4_pos", "S4_neg", "N1", "N2", "N3", "N4", "N5"]
    mu = df_eq[features].mean()
    sd = df_eq[features].std(ddof=0).replace(0, np.nan)

    z_by_cluster = {}
    top_by_cluster = {}
    for c, sub in df_eq.groupby("cluster"):
        cm = sub[features].mean()
        z = ((cm - mu) / sd).replace([np.inf, -np.inf], np.nan)
        z_by_cluster[int(c)] = z

        top = (
            z.abs()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        top_by_cluster[int(c)] = [
            {
                "feature": feat,
                "z": float(z[feat]) if pd.notna(z[feat]) else None,
                "direction": "above_city_avg" if pd.notna(z[feat]) and z[feat] > 0 else "below_city_avg",
            }
            for feat in top
        ]

    z_df = pd.DataFrame.from_dict(z_by_cluster, orient="index")
    z_df.index.name = "cluster"
    return z_df, {"top3_features_per_cluster": top_by_cluster}


def make_cluster_summary(df_eq: pd.DataFrame, *, cfg: Config) -> pd.DataFrame:
    """
    Dashboard-friendly cluster summary: counts + equity distribution + means.
    """
    grp = df_eq.groupby("cluster")
    summary = pd.DataFrame(
        {
            "n_grids_scored": grp.size(),
            "equity_mean": grp["equity_score"].mean(),
            "equity_median": grp["equity_score"].median(),
            "equity_p10": grp["equity_score"].quantile(0.10),
            "equity_p90": grp["equity_score"].quantile(0.90),
            "performance_mean": grp["performance_score"].mean(),
            "need_mean": grp["need_score"].mean(),
        }
    ).reset_index()

    summary["cluster_name"] = summary["cluster"].map(cfg.cluster_names)
    return summary


def points_to_geojson(df: pd.DataFrame, *, id_col: str = "grid_id") -> dict:
    """
    Minimal GeoJSON FeatureCollection of Point features (no geopandas needed).
    """
    feats = []
    for row in df.itertuples(index=False):
        props = row._asdict()
        lon = props.pop("lon", None)
        lat = props.pop("lat", None)
        if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
            continue
        feats.append(
            {
                "type": "Feature",
                "id": props.get(id_col),
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {k: (None if pd.isna(v) else v) for k, v in props.items()},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering + equity scoring pipeline and export dashboard artifacts.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help=(
            "Path to merged rent + 311 CSV (unit-level, same schema as DB_MVP.ipynb). "
            f"Default: Colab Drive path, or set MERGED_RENT_311_CSV. Default value: {DEFAULT_INPUT_CSV!r}"
        ),
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs.")
    parser.add_argument("--k", type=int, default=4, help="KMeans clusters (default 4).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--write-geojson", action="store_true", help="Also write GeoJSON point layers.")
    args = parser.parse_args()

    cfg = Config(k=args.k, random_state=args.random_state)
    _ensure_dir(args.output_dir)

    input_path = args.input
    if not os.path.isfile(input_path):
        if input_path != _LOCAL_CSV and os.path.isfile(_LOCAL_CSV):
            print(f"Google Drive path not accessible ({input_path!r}), falling back to local file.")
            input_path = _LOCAL_CSV
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {input_path!r}\n"
                "  • On Colab: mount Drive and ensure the file exists at the default path, or pass --input.\n"
                "  • Locally: pass your file explicitly, e.g. --input ./merged_rent_311.csv\n"
                "  • Or set environment variable MERGED_RENT_311_CSV to the full path."
            )

    grid_df = load_and_aggregate_to_grid(input_path)
    df_clust, kmeans, scaler_c = run_kmeans(grid_df, cfg=cfg)

    df_eq_base = equity_feature_engineering(df_clust, cfg=cfg)
    df_eq, meta = compute_equity_scores(df_eq_base)

    z_df, z_meta = zscore_feature_importance(df_eq)
    cluster_summary = make_cluster_summary(df_eq, cfg=cfg)

    # Join top-3 features onto scored grids (for map tooltips)
    top_map = z_meta["top3_features_per_cluster"]
    df_eq["top3_features"] = df_eq["cluster"].astype(int).map(
        lambda c: ", ".join([d["feature"] for d in top_map.get(int(c), [])])
    )

    # Exports
    grid_results_path = os.path.join(args.output_dir, "grid_results.csv")
    cluster_summary_path = os.path.join(args.output_dir, "cluster_summary.csv")
    zscores_path = os.path.join(args.output_dir, "cluster_feature_zscores.csv")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    df_eq.to_csv(grid_results_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    z_df.to_csv(zscores_path)

    _write_json(
        metadata_path,
        {
            "config": {
                "k": cfg.k,
                "random_state": cfg.random_state,
                "cluster_features": list(cfg.cluster_features),
                "service_cols": list(cfg.service_cols),
                "need_cols": list(cfg.need_cols),
                "cluster_names": cfg.cluster_names,
            },
            "pca_weights": meta["pca_weights"],
            "top3_features_per_cluster": z_meta["top3_features_per_cluster"],
            "heuristics": HEURISTICS_DEFAULT,
            "artifacts": {
                "grid_results_csv": os.path.basename(grid_results_path),
                "cluster_summary_csv": os.path.basename(cluster_summary_path),
                "cluster_feature_zscores_csv": os.path.basename(zscores_path),
            },
        },
    )

    if args.write_geojson:
        geo_cols = ["grid_id", "lat", "lon", "cluster", "equity_score", "performance_score", "need_score", "top3_features"]
        geo_df = df_eq[geo_cols].copy()
        geojson = points_to_geojson(geo_df)
        _write_json(os.path.join(args.output_dir, "grid_points.geojson"), geojson)

    # Lightweight console summary (so you can sanity-check large runs)
    print(f"Input: {input_path}")
    print(f"Aggregated grids: {len(grid_df):,}")
    print(f"Clustered grids (complete features): {len(df_clust):,}")
    print(f"Equity-scored grids: {len(df_eq):,}")
    print(f"Wrote: {grid_results_path}")
    print(f"Wrote: {cluster_summary_path}")
    print(f"Wrote: {zscores_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()

