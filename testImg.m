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
    c = floor(id/30) + 1;
    r = mod(id,30);
    if r==0
        r=30
    end
    
    r = (r-1)*20+1;
    c = (c-1)*20+1;
    
    img(r:r+19, c:c+19) = 1;
end

imshow(img);

fprintf('Testing ended\n');