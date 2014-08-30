function [cvd, cvl, trd, trl, ted, tel] = DivideData(data, label)

N = size(data,1);

cvi = floor(N/6); %cross validation end index.
cvd = data(1:cvi, :); %cross validation data.
cvl = label(1:cvi); % cross validation labels.

tri = cvi + floor(5*N/12); %train data end index.
trd = data(cvi+1:tri,:); %train data.
trl = label(cvi+1:tri); %train labels

ted = data(tri+1:end,:); %test data.
tel = label(tri+1:end);%test label.

end