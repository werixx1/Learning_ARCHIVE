<h3 align="center">Wavelet transform</h3>

  <p align="center">
    Kinda intuitive understanding, less mathy
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Explanation](#explanation)


## Explanation
> Key words: time-frequency analysis, time series data

**Wavelets** are mathematical functions that cut up data into different frequency components, and then study each component with a resolution matched to its scale (basically taken a wave they can detect different frequencies on it and place them in time, opposite to Fourier transform that doesn't consider time space when analysing waveforms, the wavelet transform provides a much richer representation that captures both **when** and **which** frequencies are present in a signal). They are very useful in situations where the signal contains **discontinuities** and **sharp spikes**.
- "In wavelet analysis, the scale that we use to look at
data plays a special role. Wavelet algorithms process data at different scales or resolutions. If we look at a signal with a large “window,” we would notice gross features. Similarly, if we look at a signal with a small “window,” we would notice small features. **The result in wavelet analysis is to see both the forest and the trees, so to speak**"
<br>
- Wavelet analysis is the breaking up of a signal into **shifted** and **scaled** versions of the original (or **mother**) **wavelet** (a mother wavelet is a specific wavelet function that is used to create a set of wavelets through shifting (translation) and scaling (dilation))
- Mother wavelet is  chosen based on the characteristics of the signal. 
Examples of wavelet types:
    ![](/resources/imgs/wavelets_4.png)
- Conditions to consider wave a wavelet:

| Zero mean   | Finite energy |
| ------------- | ------------- | 
| ![](/resources/imgs/wavelet_1.png)| ![](/resources/imgs/wavelet_2.png)

- Formula describing **continuous wavelet transform (CWT)**:
![](/resources/imgs/wavelet_3.png)
- **ψ(t)** is the mother wavelet,
- **a** is the scaling factor (dilation), which controls the frequency (simply, it stretches or compresses the wavelet),
    ![wiki_gif](https://upload.wikimedia.org/wikipedia/commons/9/95/Continuous_wavelet_transform.gif)
    > Source: Wikipedia
- **b** is the translation factor, which controls the time shift (shifts the wavelet in time, allowing us to localize where specific frequency components occur in the signal),
- **ψ** is the complex conjugate of the wavelet function.
The result of the wavelet transform is a **time-frequency representation of the signal**, showing how different frequency components evolve over time.

Scalogram analysis involves interpreting the patterns and features present in the scalogram - indentifying regions of high energy (bright spots), which indicate the presence of **significant components** at specific times and scales
- horizontal axis represents time and vertical axis represent scales (inversely related to frequency)
- the scalogram's energy distribution across scales and time provides information about signal's time-varying frequency content and the presence of transient events (short-lived) or discontinuities

| Signal    | Scalogram |
| ------------- | ------------- | 
| ![](/resources/imgs/wavelet_6.png) | ![](/resources/imgs/wavelet_7.png)
| ![](/resources/imgs/wavelet_8.png) | ![](/resources/imgs/wavelet_9.png)
 
> *(First table row)* The CWT shows nearly steady-state oscillations near 30 Hz and 37 Hz, as well as the transient events denoting the heartbeats (QRS complexes). With this analysis, you are able to analyze both phenomena simultaneously in the same time-frequency representation.
(source - [MathWorks](https://www.mathworks.com/help/wavelet/ug/practical-introduction-to-time-frequency-analysis-using-the-continuous-wavelet-transform.html))

