<h3 align="center">Timeseries analysis</h3>

  <p align="center">
    How to analyse, prepare and predict timeseries data
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Explanation](#explanation)
    - [Types of data](#types-of-data)


## Explanation
> Sources: [1](https://towardsdatascience.com/a-thorough-guide-to-time-series-analysis-5439c63bc9c5/), Geeksforgeeks, somebootcamp i cant remember its name </3

Time series data can be any data recorded in even time intervals (it's important to note that such data shouldn't be too noisy or have no patterns present). To understand how it changes over time many analysis methods are used that can help track past **patterns** and **predict** future values. By having good understanding of timeseries data patterns you can:
- predict **future trends** (like forecasting demand for a product and deciding on which months to increase production)
- detect **anomalies** in data (unusual behaviour that can ex. suggest errors in functioning of a machine in a given time)
- **predictive maintenance** (predicting potential failures of a device *before* it happens to prevent it)
- strategic planning

Usually its analysis begins with decomposing data into three components: **trend, seasonality and noise**.
![](https://blog.jetbrains.com/wp-content/uploads/2025/01/image-70.png)
- **trend** (YEAR): general tendency of data to either rise or fall, can change in different periods, but overall direction have to be visible, `the underlying direction over a long period` (up, down, flat)
- **seasonality** (MONTHS): regular and predictive flunctuations of data that tend to happen yearly in certain moths or seasons, ex. increase in sales in December, because of Chrismas (many people buying presents), it's very important to include seasonality in analysis (seasonal adjustment) and extract and understand non-seasonal data, `repeated patterns or cycles`
- **noise**: unexplained variations after removing trend and seasonality

Others:
- **cyclic patterns**: changes in time series that tend to be longer than a year + are correlated with economic cycles (like demand, goverment politics, global events)

TREND   | SEASONALITY | NOISE | CYCLE
:-------------------------:|:-------------------------: |:-------------------------: |:-------------------------:
![](/resources/imgs/ts_trend.png) | ![](/resources/imgs/ts_season.png) | ![](/resources/imgs/ts_noise.png) | ![](/resources/imgs/ts_cycle.png)

#### Types of data:
- **continious time series**: data with regular time intervals between points like sensor signals
- **discrete time series**: data collected in specyfic time periods, ex. when there are certain events we want to analyse, like daily or monthly
- **stationary**: mean, variance, pattern etc are the same over time with no trend or seasonality
- **non-stationary**: data contains trends and sesonality
- **univariate**: time series that records only ONE variable over time
- **multivariate**: time series that records MULTIPLE related variables together over time, usually they all influence each other, ex. weather prediction

#### Preprocessing time series data
- handling **missing values** 
    - ex. linear interpolation where missing value is replaced by the mean of the two known values in time series (preceding and succeeding values)
    - forward filling: filling missing values with most **recent** observed value
    - backward filling: filling missing values with **next** observed value

Missing data   | Linear interpolation | Forward filling | Backward filling 
:-------------------------:|:-------------------------: |:-------------------------: |:-------------------------:
![](/resources/imgs/ts_1.png) | ![](/resources/imgs/ts_2.png) | ![](/resources/imgs/ts_3.png) | ![](/resources/imgs/ts_4.png)
- dealing with **outliners** ()
- **stationarity and transformation**: differencing, detrending, seseasonalizing techniques to stabilize mean and variance
- **scaling and normalization**: standarization of data
for better model performance

*(to add: timeseries preprocessing techniques)*
#### Analysing techniques

Techniques usefull for detecting patterns and break series into trend, sesonality and residuals
- **autocorrelation** analysis: measuring correlation between a series and its lagged (previous) values to detect patterns
- **PACF**
- **trend** analysis: indentifying long-term direction of data
- **seasonality** analysis: detecting patterns at fixed intervals (like week, year)
- **decomposition**: separation into trend, seasonality and residual components
- **STL** (Seasonal-Trend decomposition using Loess): same as above
- **rolling correlations**: measuring correlation between data over a moving **window** 
- **ACF** and **PACF** plots  
---
#### Best approaches to time series data:
- LSTMs, LSTMs Autoencoders
- ARIMA models
- MLP Autoencoder (from keras.models import Modelautoencoder = Model(inputs=input_layer, outputs=decoder))
- Random Forests, XGBoost, One-class SVM, Bayesian Online Changepoint Detection (BOCD)
- Functional Neural Network Autoencoder
