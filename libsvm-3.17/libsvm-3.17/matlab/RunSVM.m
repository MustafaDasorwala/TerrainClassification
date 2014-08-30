%%clean
clear;
close all;

%% Load and initialize
load('data.mat'); %this data is assumed to be randomized after extraction
                         %from the image.
% data = data(1:floor(0.1*size(data,1)),:);
% data = data(randperm(size(data,1)),:);
% data = data(1:60000,:);
label = data(:,end); data(:,end) = [];
[cvd, cvl, trd, trl, ted, tel] = DivideData(data, label);

%% Perform Cross Validation
N = size(cvd,1);
K=10;
ids = crossvalind('Kfold', N, K);
errors = [];
% CArr = ([0.01,0.05, 0.1,0.5,1,5,10]); C_e=100; C_step=5;
CArr = ([0.01, 0.03, 0.04, 0.05, 0.07, 0.09]);

for ii=1:size(CArr,2)
    
    fprintf('C = %d\n', CArr(ii));
    err=0;
    for idx=1:K
        te_d = cvd(ids==idx,:);
        tr_d = cvd(~(ids==idx),:);
        te_l = cvl(ids==idx);
        tr_l = cvl(~(ids==idx));
        try
            s = svmtrain(tr_l,tr_d, ['-h 0 -t 0 -c ' num2str(CArr(ii))]);%, 'kktviolationlevel', 0.07);
        catch exception
            fprintf('C = %d didnt converge\n', CArr(ii));
            continue;
        end
        c = svmpredict(te_l, te_d, s);
        temp = (c ~= te_l);
        err = err + sum(temp)/numel(c);
        fprintf('K=%d....', idx);
    end
    errors = [errors err/K];
    fprintf('\n');
    
end
errors
[min_err,min_err_idx] = min(errors);
opti_C = CArr(min_err_idx);
fprintf('Optimal value for C = %d with error %d\n', opti_C, min_err);
plot(CArr, errors, 'bs-');
hold on;
plot(CArr(min_err_idx), min_err, 'rs');

%% Train Data
opti_C = 0.07;
fprintf('Training started....');
s = svmtrain( trd, trl);
fprintf('Training ended\n');

%% Test Data
fprintf('Testing started....');
c = svmclassify(s,ted);
temp = zeros(size(c));
temp = (c ~= te_l);
err = sum(temp)/numel(c);
fprintf('Testing ended with error = %d\n', err);