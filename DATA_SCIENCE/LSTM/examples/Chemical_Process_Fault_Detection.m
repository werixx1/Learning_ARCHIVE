%% Chemical Process Fault Detection [ using LSTM ]
%% 
% * Anomaly detection in chemical process engineering
% * Data consisting *fault free* (normal operational conditions) sensor's data 
% and *faulty data* (20 different process faults) in simulated real-world industrial 
% chemical plant 
%% 
% 
% Key Variables in the TEP Dataset and Their Importance :
% 1. Measured Variables (xmeas_1 to xmeas_41)
% 
% These variables represent real-time sensor measurements from the plant, including:
%% 
% * *Reactor temperature & pressure* (xmeas_1, xmeas_2)
% * *Separator level & pressure* (xmeas_9, xmeas_10)
% * *Flow rates of reactants and products* (xmeas_15 to xmeas_20)
% * *Composition of process streams* (mole fractions of A, B, C, etc.)
%% 
% 2. Manipulated Variables (xmv_1 to xmv_11)
% 
% These are control parameters that operators can adjust, such as:
%% 
% * *Cooling water flow rate* (xmv_2)
% * *Agitator speed* (xmv_7)
% * *Valve positions for feed streams* (xmv_5, xmv_6)
%% 
% 3. Fault Labels
% 
% The dataset includes *20 fault types*, each representing a specific anomaly:
%% 
% * *Step changes in feed composition (Fault 1-3)*
% * *Random variations in reactor cooling water (Fault 7)*
% * *Valve sticking (Fault 10)*
% * *Unknown disturbances (Fault 15-20)*
%% 
% _source:_ <https://www.linkedin.com/pulse/tennessee-eastman-process-open-source-benchmark-asma-mestaysser-k8u1f 
% The Tennessee Eastman Process: An Open-Source Benchmark for Anomaly Detection 
% in Process Engineering>
% 
% _MATLAB tutorial_: <https://www.mathworks.com/help/deeplearning/ug/chemical-process-fault-detection-using-deep-learning.html 
% Chemical Process Fault Detection Using Deep Learning - MATLAB & Simulink>
% Helper functions for preprocessing and normalizing data later on:
function processed = helperPreprocess(mydata,limit)
    H = size(mydata,1);
    processed = {};
    for ind = 1:limit:H
        x = mydata(ind:(ind+(limit-1)),4:end);
        processed = [processed; x]; %#ok<AGROW>
    end
end
%%
function data = helperNormalize(data,m,s)
    for ind = 1:size(data,1)
        data{ind} = (data{ind} - m)./s;
    end
end
% Get data
% _original source:_ <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1 
% Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation 
% - Harvard Dataverse>
url = 'https://www.mathworks.com/supportfiles/predmaint/chemical-process-fault-detection-data/faultytesting.mat';
websave('data\faultytesting.mat',url);
url = 'https://www.mathworks.com/supportfiles/predmaint/chemical-process-fault-detection-data/faultytraining.mat';
websave('data\faultytraining.mat',url);
url = 'https://www.mathworks.com/supportfiles/predmaint/chemical-process-fault-detection-data/faultfreetesting.mat';
websave('data\faultfreetesting.mat',url);
url = 'https://www.mathworks.com/supportfiles/predmaint/chemical-process-fault-detection-data/faultfreetraining.mat';
websave('data\faultfreetraining.mat',url);
%%
load('data\faultfreetesting.mat');
load('data\faultfreetraining.mat');
load('data\faultytesting.mat');
load('data\faultytraining.mat');
%%
head(faultytraining, 4)
tail(faultytraining, 4)
%% 
% * _faultNumber_ - fault type, *0* is fault free, and nums *1-20* are different 
% fault types
% * _simulationRun_ - number of times simulation ran to obtain data (nums *1-500*)
% * _sample_ - number of times TEP variables were recorded per simulation, 1 
% to 500 for the training data sets and from 1 to 960 for the testing data sets
% * _cols 4-44_ = *measured* variables of TEP
% * _45-55_ = *manipulated* variables of TEP
% remove vars with fault nums 3,9,15 - theyre not valid
faultytesting(faultytesting.faultNumber == 3,:) = [];
faultytesting(faultytesting.faultNumber == 9,:) = [];
faultytesting(faultytesting.faultNumber == 15,:) = [];
faultytraining(faultytraining.faultNumber == 3,:) = [];
faultytraining(faultytraining.faultNumber == 9,:) = [];
faultytraining(faultytraining.faultNumber == 15,:) = [];
% Exploration
f = figure;
f.Position = [100 100 1000 400];
subplot(1,4,1);
plot(faultfreetraining.xmeas_1(1:2000));
title('Fault free xmeas 1');
subplot(1,4,2);
plot(faultytraining.xmeas_1(1:2000), 'r');
title('Faulty xmeas 1');
subplot(1,4,3);
plot(faultfreetraining.xmeas_35(1:2000));
title('Fault free xmeas 35');
subplot(1,4,4);
plot(faultytraining.xmeas_35(1:2000), 'r');
title('Faulty xmeas 35');
%%
xmeas1_fault_1 = faultytraining(faultytraining.faultNumber == 1, :).xmeas_1;
xmeas1_fault_20 = faultytraining(faultytraining.faultNumber == 20, :).xmeas_20;
%%
f = figure;
f.Position = [100 100 1000 400];
subplot(1,3,1);
plot(faultfreetraining.xmeas_1(1:1000));
title('Fault free xmeas 1');
subplot(1,3,2);
plot(xmeas1_fault_1(1:1000), 'r');
title('xmeas 1 where faultNumber = 1');
subplot(1,3,3);
plot(xmeas1_fault_20(1:1000), 'r');
title('xmeas 1 where faultNumber = 20');
% Split into train/test/val sets (80/10/10)
H1 = height(faultfreetraining); 
H2 = height(faultytraining);    
msTrain = max(faultfreetraining.simulationRun)
msTest = max(faultytesting.simulationRun)      
%%
rTrain = 0.80; 
msVal = ceil(msTrain*(1 - rTrain))  
msTrain = msTrain*rTrain  
%%
sampleTrain = max(faultfreetraining.sample)
sampleTest = max(faultfreetesting.sample)
%%
rowLim1 = ceil(rTrain*H1);
rowLim2 = ceil(rTrain*H2);
trainingData = [faultfreetraining{1:rowLim1,:}; faultytraining{1:rowLim2,:}];
validationData = [faultfreetraining{rowLim1 + 1:end,:}; faultytraining{rowLim2 + 1:end,:}];
testingData = [faultfreetesting{:,:}; faultytesting{:,:}]
% Preprocess and normalize data
Xtrain = helperPreprocess(trainingData,sampleTrain);
size(Xtrain)
Ytrain = categorical([zeros(msTrain,1);repmat([1,2,4:8,10:14,16:20],1,msTrain)']);
XVal = helperPreprocess(validationData,sampleTrain);
size(XVal)
YVal = categorical([zeros(msVal,1);repmat([1,2,4:8,10:14,16:20],1,msVal)']);
 
Xtest = helperPreprocess(testingData,sampleTest);
size(Xtest)
Ytest = categorical([zeros(msTest,1);repmat([1,2,4:8,10:14,16:20],1,msTest)']);
%%
tMean = mean(trainingData(:,4:end));
tSigma = std(trainingData(:,4:end));
%%
Xtrain = helperNormalize(Xtrain, tMean, tSigma);
XVal = helperNormalize(XVal, tMean, tSigma);
Xtest = helperNormalize(Xtest, tMean, tSigma);
%%
figure;
splot = 10;    
plot(Xtrain{1}(:,1:10));   
xlabel("Time Step");
title("Training Observation for Non-Faulty Data");
legend("Signal " + string(1:splot),'Location','northeastoutside');
%%
figure;
plot(Xtrain{1000}(:,1:10));   
xlabel("Time Step");
title("Training Observation for Faulty Data");
legend("Signal " + string(1:splot),'Location','northeastoutside');
% Define model's structure and training options
numSignals = 52;
numHiddenUnits2 = 52;
numHiddenUnits3 = 40;
numHiddenUnits4 = 25;
numClasses = 18; % fault free class and all different possible process faults 
     
layers = [ ...
    sequenceInputLayer(numSignals)
    lstmLayer(numHiddenUnits2,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits3,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits4,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer];
%% 
% 
maxEpochs = 40;
miniBatchSize = 50;  
 
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize', miniBatchSize,...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ValidationData',{XVal,YVal});
% Train and evaluate network
net = trainnet(Xtrain,Ytrain,layers,"crossentropy",options);
%%
scores = minibatchpredict(net,Xtest);
Ypred = scores2label(scores,unique(Ytrain));
%%
acc = sum(Ypred == Ytest)./numel(Ypred)
%%
confusionchart(Ytest,Ypred)