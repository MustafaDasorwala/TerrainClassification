imgDataPath = '../images/data/';
imgLabelPath = '../images/labels/';
SUBSIZE=19;
trainData = dir(imgDataPath);
trainLabel = dir(imgLabelPath);
CC1=0;
CC2=0;
trainData = trainData(4:end,:);
trainLabel = trainLabel(4:end,:);
[rSize,cSize] =size(trainData);
featureVecData = [];
featureVecLbl = [];

start=51;
end_id=100;

for i = start:end_id
    fprintf('i = %d, img name = %s\n', i, trainData(i).name);
    if strcmp(trainData(i).name, 'Thumbs.db')
        continue;
    end
    try
        tData = imread([imgDataPath trainData(i).name]);
        tLabel = imread([imgLabelPath trainLabel(i).name]);
    catch
        continue;
    end
    [row,col,colorLevel] = size(tData);
    r=1;c=1;color=1;
    tData20 =0; tLabel20=0;
    rs = SUBSIZE+1;
    cs = SUBSIZE+1;
    CC1=0;
    CC2=0;
    while(col-c > 0)
        CC2=CC2+1;
        while(row-r > 0 )
            CC1=CC1+1;
            fVD = [];
            for color = 1:colorLevel
                tData20 = tData(r:r+SUBSIZE,c:c+SUBSIZE,color);
                tLabel20 = tLabel(r:r+SUBSIZE,c:c+SUBSIZE);
                fVD = [fVD imhist(tData20,10)'];
            end
            featureVecData = [featureVecData; fVD];
            fVL = imhist(tLabel20,2);
            if(fVL(2) > fVL(1))
                featureVecLbl = [featureVecLbl; 1];
            else
                featureVecLbl = [featureVecLbl; -1];
            end
            r = r+rs;
        end
    c = c+cs;
    r = 1;
    end
    featureVec = [featureVecData featureVecLbl];
end
str = ['data' num2str(start) '.mat'];
save(str, 'featureVec');
