# GoldenBall ML

## Ballon d'Or Winner Prediction System

A professional, portfolio-quality machine learning project that predicts the Ballon d'Or winner and ranks the top 10 candidates using historical voting data, player performance metrics, collective achievements, and contextual/narrative features.

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-blueviolet.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.0%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Sources](#dataset-sources)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)

## Project Overview
GoldenBall ML builds a robust, explainable machine learning system that predicts the Ballon d'Or winner and ranks the top 10 candidates. The system uses historical Ballon d'Or voting data from 1995 to present, combined with player performance metrics, team achievements, and contextual features to make accurate predictions.

## Dataset Sources
- **Historical Training Data**: Kaggle Ballon d'Or voting history datasets, RSSSF for voting results, FBref for player statistics, UEFA for Champions League performance
- **Current Season Prediction Data**: FBref, StatMuse, UEFA, WhoScored / SofaScore ratings for the current 30-man shortlist

## Methodology
- Chronological train/validation split (train on past seasons, validate on most recent completed seasons)
- Predict final voting points or rank (not just binary winner)
- Cross-validation where appropriate
- Bootstrap resampling for confidence intervals on top 10 predictions
- SHAP or permutation importance for explainability
- Multiple models tested and compared, with ensemble approach

## Models Used
*To be completed after model training and evaluation*

## Results
*To be completed after model training*

## Visualizations
*To be generated after model training*

## Installation
```bash
# Clone the repository
git clone https://github.com/chizyy7/goldenball-ml.git
cd goldenball-ml

# Install dependencies
pip install -r requirements.txt

# Run the prediction pipeline
python src/predict.py
```

## Repository Structure
```
goldenball-ml/
├── data/                    # Raw and processed data
├── notebooks/               # Jupyter notebooks for exploration and training
├── src/                     # Source code for reusable Python modules
├── models/                  # Saved trained models
├── reports/                 # Model comparison results
├── assets/                  # Generated visualizations
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── LICENSE                  # MIT License
```

## Limitations and Future Work
- Limitations to be documented after initial model development
- Future work: Incorporate real-time data updates, add more contextual features, improve explainability

## License
MIT License - see LICENSE file for details

## Credits
Built with ❤️ by chizy using Python and various ML libraries.