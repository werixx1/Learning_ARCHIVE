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
> Sources: [1](https://towardsdatascience.com/a-thorough-guide-to-time-series-analysis-5439c63bc9c5/), somebootcamp i cant remember its name </3

Time series data can be any data recorded in even time intervals. Usually its analysis begins with decomposing data into three components: trend, seasonality and noise.
![](https://blog.jetbrains.com/wp-content/uploads/2025/01/image-70.png)
- **trend** (YEAR): general tendency of data to either rise or fall, can change in different periods, but overall direction have to be visible, `the underlying direction over a long period`
- **seansonality** (MONTHS): regular and predictive flunctuations of data that tend to happen yearly in certain moths or seasons, ex. increase in sales in December, because of Chrismas (many people buying presents), it's very important to include seasonality in analysis (seasonal adjustment) and extract and understand non-seasonal data, `repeated patterns or cycles`
- **noise**: unexplained variations after removing trend and seasonality

Others:
- **cyclic patterns**: changes in time series that tend to be longer than a year + are correlated with economic cycles (like demand, goverment politics, global events)

#### Types of data:
- **continious time series**: data with regular time intervals between points
- **discrete time series**: data collected in specyfic time periods, ex. when there are certain events we want to analyse
- **stationary**: mean, variance, std etc are the same in time 
- **non-stationary**: data contains trends and sesonality

---
#### Best approaches to time series data:
- LSTMs(link), LSTMs Autoencoders
- ARIMA models
- MLP Autoencoder (from keras.models import Modelautoencoder = Model(inputs=input_layer, outputs=decoder))
- Random Forests, XGBoost, One-class SVM, Bayesian Online Changepoint Detection (BOCD)
- Functional Neural Network Autoencoder
