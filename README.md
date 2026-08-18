# README

## 1. Title and one-line summary

Code for climate attribution of power imbalance in the Yangtze River Basin during the 2022 compound drought and heatwave event (CDHE).

## 2. Project overview

This project quantifies how much the power-system imbalance was driven by the climate-change-induced CDHE in summer 2022 in the Yangtze River Basin. The analysis considers electricity cooling demand surges (Cooling Degree Days), hydropower generation deficits (Standardised Hydropower Anomaly), and power-system imbalance.

The five main provinces analysed are Sichuan, Guizhou, Hunan, Hubei, and Chongqing.

Two attribution methods are used:
1. Statistical World Weather Attribution (WWA) method.
2. HadGEM large-ensemble attribution method.

The statistical method is applied to the energy-demand case. The HadGEM-based method is applied to both the energy-demand and energy-generation cases, because the hydropower-generation data are more limited.

Main outputs include Relative Intensity Change (RIC), Risk Ratio (RR), factual and counterfactual energy-demand and energy-generation metrics, power-system imbalance under factual and counterfactual conditions, and return-period plots.

## 3. Repository contents and full code-and-data

This GitHub repository contains the complete analysis code and lightweight outputs, while the full code-and-data archive is stored on Zenodo as a compressed ZIP file. Link to the full code-and-data: https://doi.org/10.5281/zenodo.22000429

## 4. How to run

1. Download the Zenodo archive or the required data files.

2. Energy-demand attribution case:
   - Run “Yangtze_code/Demand_anomaly.ipynb” to quantify the energy-demand anomaly.
   - Run “Yangtze_code/Data_preparation_demand.ipynb” for data preprocessing, including trimming data length, ensuring resolution consistency, bias correction, and population-weighted provincial aggregation.
   - Run “Yangtze_code/GMST.ipynb” for the Method 1 statistical WWA attribution.
   - Run “Yangtze_code/Attribution_demand.ipynb” for the Method 2 HadGEM-based attribution and for visualising the demand-attribution results from Methods 1 and 2.

3. Energy-generation attribution case:
   - Run “Yangtze_code/gen_code/Gen_anomaly.ipynb” to detect the generation deficit anomaly.
   - Run “Yangtze_code/gen_code/relationship_function.ipynb” to build the relationship between Standardised Hydropower Anomaly (SHA) and Standardised Precipitation Index (SPI-6).
   - Run “Yangtze_code/gen_code/Data_preparation_gen.ipynb” for data preprocessing, including bias correction, SHA calculation from bias-corrected precipitation, and population-weighted provincial aggregation.
   - Run “Yangtze_code/gen_code/Attribution_gen.ipynb” to perform the energy-generation deficit attribution analysis using the HadGEM method.

4. Energy-imbalance analysis:
   - Run “Yangtze_code/Attribution_outage.ipynb” to quantify the change in energy imbalance under climate change.

## 5. Outputs

- Attribution tables
- RR / RIC plots
- Return-period plots
- Spatial maps
- Summary CSV files
- Factual and counterfactual power-system performance

## 6. Citation

To cite the GitHub repository:
Yang, W. (2026). Power-imbalance-climate-attribution [Code repository]. GitHub. https://github.com/VigaYang/Power-imbalance-climate-attribution

To cite the Zenodo dataset:
Yang, W. (2026). Climate attribution of power imbalance in the Yangtze River Basin during the 2022 compound drought and heatwave event (Version v1) [Data set]. Zenodo. DOI: https://doi.org/10.5281/zenodo.22000429

## 7. Contact

weijia.yang@eng.ox.ac.uk
