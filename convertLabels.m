src_path = '../images/lbl/';
dst_path = '../images/labels/';
imgs = dir(src_path);

numImgs = size(imgs,1);

for ii=3:numImgs

    imgs(ii).name
    img = imread([src_path imgs(ii).name]);
    
    r = img(:,:,1); g = img(:,:,2); b = img(:,:,3);
    idx_grey = (r==128) & (g==128) & (b==128);
    idx_orange = (r==255) & (g==128) & (b==0);
    idx = idx_grey | idx_orange;
    
    imwrite(idx, [dst_path imgs(ii).name]);
end