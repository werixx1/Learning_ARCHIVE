<h3 align="center">MATLAB syntax cheatsheet</h3>

  <p align="center">
    Data preparing, machine learning, enviroment setup
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Functions for preparing data](#functions-for-preparing-data)
- [Training a network from 'scratch'](#training-a-network-from-scratch)
    - [Setup for Experiment manager](#setup-for-experiment-manager)
    - [Steps for training a CNN network](#steps-for-training-a-cnn-network)

## Functions for EDA / data preprocessing 
> Particually for training CNN

- **Datastore** = variable for storing multiple images
```matlab
imgs_datastore = imageDatastore(path_to_images, "IncludeSubfolders", true, 
"LabelSource", "foldernames") % use folder names as labels of classes
```
- **Displaying images** from a datastore
```matlab
montage(imgs_datastore)
```

- **Imread(), imshow()** - reading a single image file as variable and displaying it
```matlab
img = imread("example.jpg")
imshow(img)
% Using empty brackets as the second input will scale the display 
% based on the minimum and maximum values present in the image.
% imshow(im,[])
```
- Extract image's **first channel** (make img gray)
```matlab
image(:,:,n) % n - channel value, 1 2 3
```
- Using **kernel** with image (kernel as a matrix)
```matlab
filtered_image = conv2(kernel, image)
% ex kernel = [ 0 1 0, 1 -4 1, 0 1 0] - blur
imshow(filtered_image, [])
% for RGB image (all channels present)
filtered_image = imfilter(image, kernel)
```
- Spliting data into **train / test**:
```matlab 
[train_df, test_ds] = splitEachLabel(imgs_datastore, proportion) % ex. 0.7 will 
% give 70/30 split
```
- Data **augmentation**:
```matlab
augmented_ds = augmentedImageDatastore([size size], ds, augmentation_type)
```
- Adjusting **learning options**
```matlab
options = trainingOptions("algorithm_name", "InitialLearnRate", learn_value)
% ex. "adam" 0.0001
```
- Return the number of non zero matrix elements (usefull when calculating **accuracy**)
```matlab
N = nnz(matrix)
% for accuracy
accuracy = nnz(test_data.Labels == predicted_labels) / length_of_data
```
- **Confusion matrix**
```matlab
confusionchart(known_labels, predicted_labels)
```
- **Predicting** values:
```matlab
minibatchpredict
```
- Inserting object **box annotation** 
```matlab
img_with_box = insertObjectAnnotation(image,...
"rectangle",box_coordinates,label)
% create a datastore of box labels
ds_of_boxes = boxLabelDatastore(table)
```
![alt text](/resources/imgs/MATLAB_3.png)
(labels for these objects here are 4 and 6, but usually would be categories names)

- **Plots next to each other**
  - subplot(x,y,z): x - rows, y - columns, z - position of the subplot on the figure
```matlab
figure;
subplot(1,2,1);
plot(...);
subplot(1,2,2);
plot(...);
```

- Find rows of a table which **meet a specific condition**
```matlab
results = table_name(table_name.column == value, :);
```
(or use **all data()**)

## Training a network from 'scratch' 
Example with training a simple CNN for image classification
### Setup for Experiment manager 
**.mat (Setup Function)**
- In Expirement Manager:
![matlab_1](/resources/imgs/MATLAB_1.png)
    - ^Generates a script to define training data, layers and training options
- Adjusting hyperparameters:
![matlab_2](/resources/imgs/MATLAB_2.png)
- And to call them in workspace (to view or change) type:
    ```matlab
    params.myInitialLearnRate
    params.myEpochs
    ```
### Steps for training a CNN network
Function that loads defined params into training 
> Simplified steps, assuming data is already preprocessed and labeled

(It's generated automatically after setting up everything in Experiment manager)
```matlab
function [xTrain,yTrain,layers,options] =
Expirement_setup1(params) % has all parameters from below
% to load into experiment manager ? i think ?
```
- **Load** training data (expecting its already divided into train/test):
```matlab
data = load("file_name.mat");
yTrain = data.yTrain;
yTest = data.yTest;
% numer - num of array elements (len())
xTrain = reshape(data.xTrain, 28,28,1, numel(yTrain));
% reshaping so it matches network input layer
xTest = reshape(data.xTest, 28,28,1, numel(yTest));
% for images = imresize()
```
[Train\test\val for labeled data]
```matlab
idx = splitlabels(labels,[0.9,0.05]) % train test val split
```
- Define network **architecture**
```matlab
layers = [
imageInputLayer([28 28 1])
convolution2dLayer(3,8,"Padding","same")
batchNormalizationLayer
reluLayer
% etc
]
```
- Specyfy **training options**
```matlab
options = trainingOptions("adam", ...
"InitialLearnRate",params.myInitialLearnRate,...
"MaxEpochs",params.myEpochs, ...
"ValidationData",{xTest yTest}, ...
"ValidationFrequency",200,...
"Shuffle","every-epoch",...
"Verbose",false);
end
```