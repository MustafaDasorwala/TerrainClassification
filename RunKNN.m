%% --------------- Clean the space------------------
clear;
close all;

%% Load and initialize
load('data.mat'); %this data is assumed to be randomized after extraction
                         %from the image.
data = data(1:12000,:);
label = data(:,end); data(:,end) = [];
[cvd, cvl, ted, tel, trd, trl] = DivideData(data, label);

%% Perform Cross Validation
N = size(cvd,1);
Kfold=10;
ids = crossvalind('Kfold', N, Kfold);
errors = [];
K_s=1; K_e=50; K_step=5;

for K = K_s:K_step:K_e
    
    fprintf('K = %d\n', K);
    err=0;
    for idx=1:Kfold
        te_d = cvd(ids==idx,:);
        tr_d = cvd(~(ids==idx),:);
        te_l = cvl(ids==idx);
        tr_l = cvl(~(ids==idx));
        
        class_l = knnclassify(te_d,tr_d,tr_l,K);
        err = err + sum((class_l ~= te_l))/size(te_l,1);
        fprintf('K=%d....', idx);
    end
    errors = [errors err/Kfold];
    fprintf('\n');
    
end
K_plot = K_s:K_step:K_e;
errors
[min_err,min_err_idx] = min(errors);
opti_K = K_s + (min_err_idx-1)*K_step;
fprintf('Optimal K = %d with error %d\n', opti_K, min_err);
knnFig = figure;
hold on;
plot(K_plot, errors, 'bs-');
plot(opti_K,min_err, 'rs');

%% Train Data
fprintf('Training started....');
classl = knnclassify(ted,trd,trl,opti_K);
fprintf('Training ended\n');

%% Test Data
fprintf('Testing started....');
err = sum(classl ~= tel)/numel(tel);
fprintf('Testing ended with error = %d\n', err);

%% Storing Data
KnnData.Ks = K_plot;
KnnData.cvErrors = errors;
KnnData.OptimumK = opti_K;
KnnData.TestError = err;

save('KnnResults','KnnData');
