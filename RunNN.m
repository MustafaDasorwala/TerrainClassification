%% clear
load('data.mat');
label = data(:,end);
data(:,end) = [];
target = zeros(numel(label), 2);
target(label==-1,1) = 1;
target(label==1,2) = 1;
data = data';
target = target';

%% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.valRatio = 1/6;
net.divideParam.trainRatio = 5/12;
net.divideParam.testRatio = 5/12;
net.performFcn

% Train the Network
[net,tr] = train(net,data,target);

%% Test the Network
outputs = net(data(:,70001:120000));
errors = gsubtract(target(:,70001:120000),outputs);
% performance = perform(net,target,outputs)

%% View the Network
view(net)