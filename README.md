# Bangalore Illegal Parking Spots — Gridlock Intelligence

AI-driven parking hotspot detection, congestion impact scoring, and enforcement prioritization for Bangalore.

## Overview

**Bangalore Illegal Parking Spots — Gridlock Intelligence** is a spatial analytics and forecasting system that transforms parking violation records into actionable enforcement intelligence. The project is built around a simple but important idea:

> Not all parking violations create the same amount of congestion.

A small cluster of vehicles blocking a junction can be far more disruptive than a larger number of vehicles parked in a low-impact side street. This repository turns that principle into a measurable, repeatable workflow for city-scale enforcement planning.

The solution combines:

- **H3 spatial aggregation** for consistent city-wide analysis
- **source-based vehicle equivalency values** for impact-aware scoring
- **spatial hotspot detection** with Getis-Ord Gi*
- **time-bucketed analysis** to reflect patrol shifts and traffic patterns
- **forecasting** with XGBoost
- **micro-hotspot discovery** with DBSCAN
- **interactive dashboards** for historical and predictive review

## Problem Statement

### Poor Visibility on Parking-Induced Congestion

On-street illegal parking and spillover parking near commercial areas, metro stations, and events choke carriageways and intersections.

### Why It’s Hard Today

- Enforcement is patrol-based and reactive.
- There is no heatmap of parking violations vs. congestion impact.
- It is difficult to prioritize enforcement zones.

### Problem Statement Direction

**How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?**

## Our Approach

This solution addresses the problem with a multi-layer intelligence pipeline:

1. **Aggregate raw violations into spatial cells** using H3 hexagons.
2. **Score each cell by both volume and traffic impact** using a Congestion Impact Score.
3. **Detect spillover hotspots** with Local Getis-Ord Gi* statistics.
4. **Classify enforcement priority** into operational tiers.
5. **Forecast future hotspots** using XGBoost-based hourly predictions.
6. **Simulate micro-clusters** with DBSCAN to expose localized enforcement clusters.
7. **Present everything in an interactive Streamlit dashboard** for real-world decision making.

This turns raw violation data into a live command-center style view for traffic enforcement teams.

## Key Capabilities

- **Illegal parking hotspot mapping** at H3 resolution 8
- **Congestion Impact Score (CIS)** for severity-aware prioritization
- **Spatial spillover detection** with Local Getis-Ord Gi*
- **Strategic enforcement tiers** to guide patrol allocation
- **7-day forecast of violation density and congestion impact**
- **DBSCAN clustering** for micro-hotspot discovery
- **Interactive dashboard** with historical and predictive layers
- **Downloadable forecast exports** for field teams and planning

## Methodology and Decision Rationale

### 1) Why H3 hexagons

We use H3 because enforcement is operationally zone-based, not point-based. Hexagonal cells provide a stable spatial unit for ranking, comparison, and neighborhood analysis. Compared with square grids, hexagons reduce directional bias and provide a more natural adjacency structure for spillover analysis.

### 2) Why H3 resolution 8

Resolution 8 is a practical city-scale compromise. It is fine enough to expose street-level differences, but not so fine that the dashboard becomes noisy or difficult to interpret for command-center use. It also maps well to patrol planning, where decisions are usually made at beat or zone level rather than at a single coordinate.

### 3) Why source-based vehicle scores instead of heuristics

Vehicle impact is measured using passenger car equivalent values sourced from **IRC:106-1990** and related Indian traffic engineering references, rather than hand-tuned weights. This matters because the traffic mix in Indian cities is heterogeneous, with a large share of two-wheelers, autos, cars, commercial vehicles, and buses sharing the same road space. Source-based scores are more defensible, reproducible, and aligned with established traffic engineering practice than arbitrary heuristic weights.

### 4) Why location multipliers are used

A parked vehicle does not have the same operational impact everywhere. Parking near a junction, intersection, signal, zebra crossing, or arterial road creates more friction than parking on a low-volume side street. The location factor is included so the score reflects both **what** is parked and **where** it is parked.

### 5) Why Getis-Ord Gi* is used

Raw counts alone do not reveal whether violations are spatially clustered enough to represent a real hotspot. Gi* adds the neighborhood context needed to distinguish isolated high-count cells from meaningful spillover clusters. This makes the hotspot map statistically grounded rather than purely descriptive.

### 6) Why time buckets are used

Parking behavior and enforcement visibility vary by time of day. Discretizing the data into time buckets makes the output actionable for shift-based deployment and avoids smoothing over important operational differences. It also helps expose enforcement bias, such as missing or underrepresented peak periods.

### 7) Why H3 polygon layers are used for the dual heatmap

The dual heatmap is designed as a direct comparison between two comparable layers:

- **Layer A:** violation density / impact per H3 cell
- **Layer B:** hotspot significance per H3 cell

Using H3 polygons ensures both layers share the same boundaries. That makes the comparison precise. A cell that is strong in Layer A but absent in Layer B is immediately visible as a high-volume, low-significance zone; a cell that is strong in Layer B but weak in Layer A becomes an under-enforced high-impact zone.

### 8) Why XGBoost is used for forecasting

The forecasting task is tabular and feature-rich: time lags, rolling statistics, spatial identity, time-of-day flags, and historical patterns all matter. XGBoost is well suited to this kind of structured data and provides strong performance without requiring a heavy sequence model.

### 9) Why a naive baseline matters

Forecasting should be judged against a simple reference such as a 7-day lag baseline. Without a baseline, it is hard to tell whether the model is actually adding value. Comparing against a naive predictor makes the forecast evaluation honest and interpretable.

### 10) Why DBSCAN is used

H3 cells are useful for city-wide operations, but some enforcement problems require a finer lens. DBSCAN helps detect dense micro-clusters of illegal parking points, such as double-parking pockets or localized spill zones around junctions.


## How We Tackled the Problem in Detail

## 1) Spatial intelligence

The core challenge in parking enforcement is not just counting violations. The real challenge is understanding **where violations concentrate**, **when they intensify**, and **how much they disrupt flow**.

To solve this, the raw violation records are first cleaned and standardized in `preprocess_and_train.py`:

- invalid coordinates are removed
- timestamps are converted from UTC to IST
- date, hour, weekday, month, and week-of-year are derived
- each record is mapped to an **H3 hex cell** at resolution 8
- time-of-day buckets are created for operational reporting
- violation categories are parsed into a primary violation type
- vehicle types are normalized into major classes

This gives a consistent city-wide spatial grid that can be analyzed over time.

## 2) Measuring impact

A major weakness in traditional enforcement is that it treats all violations as equal. This project fixes that by assigning each violation a **Passenger Car Equivalent (PCE)** weight and a location multiplier.

### Congestion Impact Score (CIS)

The repository implements a weighted approach in `preprocess_and_train.py`:

- two-wheelers contribute less than large vehicles
- buses and heavy vehicles contribute much more
- violations near junctions, signals, intersections, zebra crossings, and arterial roads are amplified through a location factor

This produces a **Congestion Impact Score** that reflects actual traffic friction instead of raw ticket count.

In operational terms:

- a scooter on a side lane is not treated the same as a bus at an intersection
- silent bottlenecks become visible
- enforcement can focus on the cells that matter most for movement, not just the cells with the most tickets

## 3) Building a city-wide hourly hotspot model

After aggregation, the data is converted into **H3 cell × date × hour** buckets. This enables the system to learn patterns at a granularity suitable for traffic enforcement.

The model engineering pipeline includes:

- **lag features**: 1h, 2h, 3h, 24h, 168h
- **rolling features**: 3h, 6h, 24h, 7d means and 24h std dev
- **cyclical time encodings**: sine/cosine for hour, weekday, and month
- **operational flags**: weekend, morning, peak hour, night
- **spatial features**: H3 cell id, cell coordinates, historical mean
- **composition features**: vehicle mix and violation mix

These features let the model capture:

- time-of-day effects
- weekly recurrence
- cell-specific behavior
- historical persistence of congestion
- change in pattern by location type

The system trains two XGBoost models:

- a **regressor** for predicted violation count
- a **classifier** for hotspot probability

The classification target is hotspot activity at **10 violations per hour**.

This is a practical threshold for planning enforcement intensity.

## 4) Detecting spillover and neighborhood effects

Parking problems rarely stay inside a single street segment. One blocked road often causes congestion on adjacent streets as vehicles divert, queue, or slow down.

To measure this, `scoring.py` computes **Local Getis-Ord Gi*** statistics using:

- an H3 neighbor graph built from 1-ring adjacent cells
- row-standardized spatial weights
- full city reindexing so that the statistical background is consistent

This creates a **spatial hot-spot score** for every cell.

### What this gives the enforcement team

- detection of clustered congestion rather than isolated points
- visibility into spillover zones near chokepoints
- a statistically grounded way to identify neighborhood-wide impact

This is important because a low-volume cell may still be a major problem if it sits inside a cluster of high-impact cells.

## 5) Turning analytics into enforcement tiers

The project does not stop at visualization. It converts analytics into **actionable enforcement classes**.

In `scoring.py`, cells are classified into:

- **Tier 1: Max Disruption**
  - high violation volume
  - high congestion impact
  - highest operational priority

- **Tier 2: Silent Bottleneck**
  - lower violation volume
  - high congestion impact
  - critical because these are hidden flow blockers

- **Tier 3: Volume Hotspot**
  - high violation volume
  - lower congestion impact
  - useful for routine ticketing

- **Standard**
  - remaining cells

The thresholds are based on the **90th percentile of active cells**, which makes the classification adaptive to the city’s real distribution rather than relying on arbitrary fixed cutoffs.

This is particularly useful for dispatching:

- towing vehicles
- enforcement patrols
- targeted monitoring teams

## 6) Forecasting future enforcement demand

The repository also generates a **7-day hourly forecast** in `preprocess_and_train.py`.

The forecast pipeline:

- trains XGBoost models on historical cell-hour behavior
- predicts violation counts for each H3 cell and future hour
- predicts hotspot probability
- estimates future congestion impact
- saves forecast results to `predictions_7d.parquet`

To make the forecast more realistic, `prediction.py` applies **temporal calibration** based on historical per-cell hourly and day-of-week behavior. This avoids a flat-looking forecast where adjacent hours and days are treated the same.

In practice, this means the system does not merely project a static average into the future. It reintroduces the natural rhythm of Bangalore traffic patterns.

## 7) Finding micro-hotspots with DBSCAN

While H3 aggregation is excellent for city-scale planning, some problems require a finer lens.

`data_processing.py` includes DBSCAN clustering on raw violation coordinates using haversine distance. This helps identify:

- localized double-parking pockets
- footpath parking bursts
- dense roadside micro-clusters
- clustered spill locations around sensitive junctions

The dashboard can run clustering for both historical data and simulated forecast points, helping enforcement teams understand the **shape** of a hotspot, not just its aggregate severity.

## System Architecture

### Pipeline Flow

1. **Raw violation data**
2. **Cleaning and enrichment**
3. **H3 aggregation and feature engineering**
4. **Model training with XGBoost**
5. **Historical cache generation**
6. **7-day forecast generation**
7. **Spatial hotspot scoring with Gi***
8. **Enforcement tier classification**
9. **Streamlit dashboard for exploration and planning**

### Main Components

- `preprocess_and_train.py` — end-to-end preprocessing, feature engineering, training, and artifact generation
- `data_processing.py` — data loading, filtering, and DBSCAN clustering helpers
- `scoring.py` — spatial weights, Gi* computation, and ROI classification
- `prediction.py` — forecast loading, calibration, and filtering helpers
- `app.py` — Streamlit UI for historical and predictive analysis

## Dashboard Views

### Historical Analysis

- Violation density heatmap
- Congestion impact heatmap
- Combined density + CIS overlay
- Spillover hotspot map
- Strategic enforcement plan
- DBSCAN clustering of historical violation points

### Predictive Forecasting

- Forecasted violation density
- Forecasted congestion impact
- Combined predicted overlay
- Predicted spillover hotspots
- Predicted enforcement plan
- Predicted DBSCAN clusters
- Model metrics and feature summary

## Technical Stack

- **Python**
- **Streamlit** for the dashboard
- **Pandas / NumPy** for data processing
- **H3** for spatial indexing
- **XGBoost** for forecasting and classification
- **scikit-learn** for preprocessing and DBSCAN
- **PyDeck** for interactive map rendering
- **libpysal + esda** for spatial statistics
- **Matplotlib** for charts
- **PyArrow** for Parquet caching

## Repository Structure

```text
.
├── app.py
├── data_processing.py
├── prediction.py
├── preprocess_and_train.py
├── scoring.py
├── requirements.txt
├── solution.py
├── parking_hotspot_prediction (2) (1).py
└── scratch/
    ├── test_coords.py
    └── test_minimal_map.py

## Requirements

See `requirements.txt`.

Core dependencies include:

- streamlit
- pydeck
- h3
- pandas
- numpy
- matplotlib
- pyarrow
- scikit-learn
- xgboost
- folium
- branca
- libpysal
- esda

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare model artifacts

Run the preprocessing and training pipeline:

```bash
python preprocess_and_train.py
```

This generates the cached artifacts used by the app:

- `cell_metadata.parquet`
- `historical_aggregated.parquet`
- `violations_dbscan.parquet`
- `predictions_7d.parquet`
- `xgb_regressor.pkl`
- `xgb_classifier.pkl`
- `model_meta.pkl`

### 3. Launch the dashboard

```bash
streamlit run app.py
```

---

## Why This Solution Is Operationally Useful

This project is designed for traffic enforcement teams, city planners, and mobility analysts who need to answer questions like:

- Where are the worst illegal parking hotspots?
- Which locations actually slow traffic the most?
- Which areas should receive patrols first?
- Which hotspots are likely to worsen next week?
- Which clusters represent silent but serious bottlenecks?

The system provides a **data-driven enforcement priority map**.

---

## Features

- Uses a **multi-signal risk model** instead of raw counts alone
- Combines **spatial analytics, ML forecasting, and operational tiering**
- Preserves **historical and predictive views** in one interface
- Supports **downloadable operational outputs** for field planning
- Applies **statistical significance testing** for hotspot detection
- Uses **city-scale spatial partitioning** via H3 for consistency
- Distinguishes **volume hotspots** from **traffic-critical bottlenecks**

---

## Important Notes

- This repository expects the cached Parquet and model artifacts generated by `preprocess_and_train.py`.
- The app is optimized around Bangalore traffic enforcement use cases, but the approach can be adapted to other cities with similar violation data.

---

## Future Enhancements

- Live integration with municipal enforcement feeds
- Real-time congestion proxy signals from traffic APIs
- Route-level impact estimation
- Multi-resolution H3 analysis
- Explainability layer for model-driven enforcement decisions
- Automated patrol recommendation engine
- Mobile-friendly field officer view

---

## Acknowledgment

Built for the Bangalore illegal parking and gridlock enforcement problem, with the goal of helping cities move from reactive patrols to proactive, intelligence-led enforcement.
