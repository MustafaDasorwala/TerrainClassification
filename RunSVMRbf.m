%% Load and initialize
load('data.mat'); %this data is assumed to be randomized after extraction
                         %from the image.
%data = data(randperm(size(data,1)),:);
data = data(1:12000,:);
label = data(:,end); data(:,end) = [];
[cvd, cvl, ted, tel, trd, trl] = DivideData(data, label);
%% Perform Cross Validation
N = size(cvd,1);
K=10;
ids = crossvalind('Kfold', N, K);
errors = [];
CArr = ([0.01,0.1,1,2,5,10]);

for ii=1:size(CArr,2) 
    
    fprintf('C = %d\n', CArr(ii));
    err=0;
    for idx=1:K
        te_d = cvd(ids==idx,:);
        tr_d = cvd(~(ids==idx),:);
        te_l = cvl(ids==idx);
        tr_l = cvl(~(ids==idx));
        
        s = svmtrain(tr_d,tr_l,'boxconstraint',CArr(ii),'kernel_function','rbf');
        c = svmclassify(s,te_d);
        temp = zeros(size(c));
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
fprintf('Training started....');
s = svmtrain( trd, trl,'boxconstraint',opti_C,'kernel_function','rbf');
fprintf('Training ended\n');

%% Test Data
fprintf('Testing started....');
c = svmclassify(s,ted);
temp = zeros(size(c));
temp = (c ~= tel);
err = sum(temp)/numel(c);
fprintf('Testing ended with error = %d\n', err);

%% Storing Data
SVMrbf.CArr = CArr;
SVMrbf.cvErrors = errors;
SVMrbf.OptimumC = opti_C;
SVMrbf.TestError = err;

save('SVMrbfResults','SVMrbf');
