# AI-Driven Parking Intelligence for Bengaluru

This project was built to address **poor visibility on parking-induced congestion** in Bengaluru.

## Problem Context

### Operational Challenge
On-street illegal parking and spillover parking near commercial areas, metro stations, and event zones choke carriageways and intersections.

### Why This Is Hard Today
- Enforcement is mostly patrol-based and reactive.
- There is no clear heatmap linking parking violations to congestion impact.
- It is difficult to prioritize where enforcement should be focused first.

## Problem Statement
**How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?**

## Project Goal
Build a data-driven intelligence layer that:
- Identifies likely illegal parking hotspots.
- Estimates congestion impact in nearby road cells/intersections.
- Produces actionable hotspot maps and risk scores for enforcement teams.

## Repository Overview
- `app.py` - Streamlit app for visualization and interaction.
- `data_processing.py` - Data preparation and feature engineering utilities.
- `preprocess_and_train.py` - Model training pipeline.
- `prediction.py` - Hotspot/congestion prediction workflows.
- `scoring.py` - Scoring logic for model outputs.
- `solution.py` - Supplementary solution script/notebook-style content.

## Data & Artifacts (present in repo)
- Historical and processed parquet datasets for violations and aggregated traffic context.
- Trained model artifacts (`.pkl`) for classification/regression.
- Generated prediction outputs (`predictions_7d.parquet`).

## High-Level Approach
1. Ingest and clean violation + traffic context data.
2. Engineer spatial-temporal features.
3. Train models to predict hotspot likelihood and congestion impact.
4. Generate hotspot intelligence outputs for targeted enforcement planning.

## Expected Outcome
A practical decision-support system that helps city teams move from reactive patrolling to proactive, impact-based enforcement.
