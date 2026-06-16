# NORW010 tree-ring preprocessing for the Kalman filtering experiment

This folder contains small reproducibility scripts for the real-world dendrochronological example used in the Kalman filtering experiment. The scripts prepare ring-width index matrices from raw tree-ring measurements of Scots pine from the Forfjorddalen site in Nordland, northern Norway.

The dataset is publicly available from the NOAA Paleoclimatology archive under the identifier NORW010 and is associated with the coastal northern Norway temperature reconstruction by Kirchhefer. The raw dataset contains 71 individual ring-width measurement series from 36 Scots pine trees and spans 877-1994 CE.

All executable scripts are located in the `scripts` directory. The `Download_rwl.R` script downloads the source NORW010 files from the NOAA Paleoclimatology archive and saves them to the `data` directory. The `preprocessingtrw.py` script performs the simple Python preprocessing pipeline and writes its output to `results/python`. The `Preprocess_rwl.R` script performs the conventional R/dplR preprocessing pipeline and writes its output to `results/dplr`.

The `examples` directory contains a Jupyter notebook version of the Python preprocessing script.

## What the scripts do

The Python script implements the deliberately simple preprocessing used before the proposed Kalman filtering procedure. Individual raw series are aligned by age, a mean age-growth curve is estimated across the available series and smoothed using a 51-year centered moving window, and each raw series is divided by the corresponding age-specific mean value.

The R script implements a conventional dendrochronological baseline using the `dplR` package. The raw observations are subjected to Cook's power transformation, age-dependent spline detrending with a 50-year parameter, and signal-free standardization. 

## How to run

### using the RGui

**In RGui, commands are entered without `Rscript`. First set the working directory to the project root:**
setwd("C:/Project")

**Install `dplR` if it has not been installed yet:**
install.packages("dplR", repos = "https://cloud.r-project.org")

**Download the NORW010 source files into the `data` directory:**
source("scripts/Download_rwl.R", encoding = "UTF-8")

**Run the R/dplR preprocessing pipeline:**
source("scripts/Preprocess_rwl.R", encoding = "UTF-8")

### using the Jupyter

**The `examples` directory contains a notebook version of the Python preprocessing workflow:**
cd /d "C:\Project"

**Python packages:**
python -m pip install numpy pandas matplotlib

**If Jupyter is not installed, it can be installed with:**
python -m pip install notebook

**Start JupyterLab**
jupyter lab

