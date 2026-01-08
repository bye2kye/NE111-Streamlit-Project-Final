## Description

This project is an interactive Streamlit application designed to bridge the gap between raw numerical data and meaningful statistical insight. It enables users to quickly visualize data distributions without manual preprocessing or cleanup.

The tool supports dual data ingestion through free-form text input (comma- or space-separated values) and direct CSV file uploads. Input data is automatically sanitized, parsed, and converted into floating-point arrays for analysis.

Dynamic histograms are generated using Matplotlib with a custom dark-mode theme for high-contrast, presentation-ready visuals. Users can fit a variety of probability distributions (including Normal, Gamma, Weibull, Lognormal, and others) using SciPy, with both automatic parameter estimation and manual parameter tuning via interactive sliders.

Quantitative error metrics, including mean squared error (MSE) and maximum absolute error, are computed to evaluate the quality of fitted distributions relative to the histogram data. The interface updates reactively, allowing real-time exploration of distribution behavior and parameter sensitivity.
