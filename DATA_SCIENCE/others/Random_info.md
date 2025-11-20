"You count **discrete** data. You measure **continuous** data"
![data-type](/resources/imgs/RANDOM_1.png)

--- 
### ML pipeline:
1. Step one is always **collecting** and **loading** the data into workspace. Depending on datatype you may need different approaches, but in general it's best practice to have it sorted and labeled.
2. **Understand your data**, explore - look for correlation between variables, check mean, most common values etc
3. Data **preprocessing**. This process can look like these:
    - cutting outliners,
    - normalization (scaling data),
    - checking for missing values and choosing substitution approach,
    - resampling data to regular time intervals,
    - rezising images to match the same size and reducing dimensionality,
    - PCA (for reducing features),
    - augmentation,
    - and many more depending on your data
4. **Feature extraction** - turning raw data into information that can be used for model training. Depending on data type (numer, images, sensor data etc) the process looks differently. Some examples:
    - images: extracting edge locations, resolution, colors
    - sensor data: extracting signal properties from peak analysis, spectral measurements etc
5. Divide data into **train/test** (+validation) sets.
6. Choose **model type** for the task, create it's **architecture** (layers) and choose training options.
7. **Train** and **test** the model.
8. **Improve** the model if needed. Examples:
    - reducing features
    - different training options (optimizer, epochs, batch size etc)
    - reshaping data
    - adding / reducing complexity (amount of neurons, amount of layers, different activation functions in hidden layers, etc)
    - techniques like early stopping and similar
---