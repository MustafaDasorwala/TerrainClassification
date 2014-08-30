%% --------------- Clean the space------------------
clear;
close all;

%% Load and initialize
load('data.mat'); %this data is assumed to be randomized after extraction
                         %from the image.
data = data(1:12000,:);
label = data(:,end); data(:,end) = [];

% data = load('train_data.mat'); 
% label = load('train_labels.mat');
% data = data.train_data;
% label = label.train_labels;

% N = size(data,1);
% d = size(data,2);
% 
% cvi = floor(N/6); %cross validation end index.
% cvd = data(1:cvi, :); %cross validation data.
% cvl = label(1:cvi); % cross validation labels.
% 
% tri = cvi + floor(5*N/12); %train data end index.
% trd = data(cvi+1:tri,:); %train data.
% trl = label(cvi+1:tri); %train labels
% 
% ted = data(tri+1:end,:); %train data.
% tel = label(tri+1:end);

[cvd, cvl, ted, tel, trd, trl] = DivideData(data, label);

%% Perform Cross Validation
N = size(cvd,1);
K=10;
ids = crossvalind('Kfold', N, K);
errors = [];
nTree_s=50; nTree_e=100; nTree_step=10;

for nTrees = nTree_s:nTree_step:nTree_e
    
    fprintf('num of trees = %d\n', nTrees);
    err=0;
    for idx=1:K
        te_d = cvd(ids==idx,:);
        tr_d = cvd(~(ids==idx),:);
        te_l = cvl(ids==idx);
        tr_l = cvl(~(ids==idx));
        
        b = TreeBagger(nTrees, tr_d, tr_l);
        predl = char(b.predict(te_d));
        predl = str2num(predl);
        err = err + sum((predl == te_l)~=1)/numel(te_l);
        fprintf('K=%d....', idx);
    end
    errors = [errors err/K]
    fprintf('\n');
    
end
errors
[min_err,min_err_idx] = min(errors);
opti_nTrees = nTree_s + (min_err_idx-1)*nTree_step;
fprintf('Optimal num of trees = %d with error %d\n', opti_nTrees, min_err);
Ns = nTree_s:nTree_step:nTree_e;
hold on;
plot(Ns, errors, 'bs-');
plot(opti_nTrees,min_err, 'rs');

%% Train Data

opti_nTrees = 70;
fprintf('Training started....');
b = TreeBagger(opti_nTrees, trd, trl);
fprintf('Training ended\n');

%% Test Data
fprintf('Testing started....');
predl = char(b.predict(ted));
predl = str2num(predl);
err = sum((predl == tel)~=1)/numel(tel);
fprintf('Testing ended with error = %d\n', err);

%% Storing Data

RFData.Ns = Ns;
RFData.cvErrors = errors;
RFData.OptimumN = opti_nTrees;
RFData.TestError = err;

save('RFResults','RFData');


%% Test Image
fprintf('Testing started....');
i_d = featureVec(:,1:30);
i_l = featureVec(:,end);
predl = char(b.predict(i_d));
predl = str2num(predl);
img = zeros(600,800);
idx = find(predl==1);

% generate labelled image
for ii=1:numel(idx)
    id=idx(ii);
    r = floor(id/30) + 1;
    c = mod(id,30);
    
    img(r:r+19, c:c+19) = 1;
end

imshow(img);

fprintf('Testing ended\n');