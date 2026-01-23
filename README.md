# Forecast Evaluation Package

A Python package for analysing and visualising economic forecast data.

## Installation

```sh
# install from PyPI
pip install forecast_evaluation
```

## Documentation

The package documentation can be found [here](https://bank-of-england.github.io/forecast_evaluation/) with examples on how
to use the package in this [notebook](https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/example_notebook.ipynb).


## Features
The package contains tools to inspect the accuracy, unbiasedness and efficiency of economic forecasts. It includes visualisation tools, statistical tests and accuracy metrics commonly used for forecast evaluation. In handles both point forecasts (with the `ForecastData` object) and density forecasts (with the `DensityForecastData` object).

### Visualisation for forecasts, outturns and errors:
* Forecast vintages plot
* Accuracy and bias plots (average and rolling averages)
* Hedgehog plots
* Outturn revisions
* Forecast error distributions

### Statistical tests
* Accuracy analysis (Diebold-Mariano test)
* Bias analysis (Mincer-Zarnowitz Regression)
* Weak Efficiency analysis (Revision predictability)
* Strong Efficiency analysis (Blanchard-Leigh regression)
* Testing correlation between forecast revisions and forecast errors
* Rolling-window analysis of most tests with fluctuation tests.

### Accuracy metrics available
* Root mean square error
* Mean absolute error
* Median absolute error

All of the above features can be explored interactively in a dashboard.

## Loading data

### Data format
The forecasts should be in a `pandas` dataframe format with the following structure:
```
            date vintage_date variable       source frequency  forecast_horizon  value
0     2014-12-31   2015-03-31      gdp        BVAR         Q                 0    100
1     2015-03-31   2015-03-31      gdp        BVAR         Q                 1    101
2     2015-06-30   2015-03-31      gdp        BVAR         Q                 2    102
3     2015-09-30   2015-03-31      gdp        BVAR         Q                 3    103
```

Outturns follow the same structure but do not contain a `source` column. 

### Creating a ForecastData instance
The package's main object is the `ForecastData` class which holds the outturns, forecasts, transformed forecasts and forecast errors. You can create an instance of this class with:

```python
import forecast_evaluation as fe

forecast_data = fe.ForecastData(forecasts=forecasts_dataframe, outturns=outturns_dataframe)
```

The package also comes with built-in data used in the Bank of England 2026 Forecast Evaluation Report which can be loaded with:
```python
forecast_data = fe.ForecastData(load_fer=True) 
```

The forecast_data object has methods to filter, analyse and visualise the data and resulting analysis. These are illustrated in the [example notebook](https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/example_notebook.ipynb).

Results from the Bank of England 2026 Forecast Evaluation [Macro Technical Paper](https://www.bankofengland.co.uk/macro-technical-paper/2026/learning-from-forecast-errors-the-banks-enhanced-approach-to-forecast-evaluation) can also be replicated with [this notebook](https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/mtp_replication_notebook.ipynb).


## Run the dashboard
To make visualisation of forecasts and their properties easier, the package includes a dashboard. Once a ForecastData object has been created the dashboard can be run with:
```python
forecast_data.run_dashboard()
```
