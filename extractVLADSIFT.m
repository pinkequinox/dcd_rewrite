function [ vec ] = extractVLADSIFT( img_mb, img_mm, class_num )

num_mb = length(img_mb);
num_mm = length(img_mm);
%% get all sift and save
if ~exist('allsift.mat'))
    allsift = [];

    for m = 1:num_mb
        [f, d] = vl_sift(img_mb{m});
        d = im2double(d);
        allsift = [allsift; d'];
    end

    for m = 1:num_mm
        [f, d] = vl_sift(img_mm{m});
        d = im2double(d);
        allsift = [allsift; d'];
    end

    save allsift.mat allsift
end
%% 

% ================================================================
% VLAD
if class_num == 0
   class_num = 4;
end
load('allsift.mat');
allsift = double(allsift);
centers = vl_kmeans(allsift', class_num);
kdtree = vl_kdtreebuild(centers);
for m = 1:num_mb
    [f, d] = vl_sift(img_mb{m});
    d = im2double(d);
    nn = vl_kdtreequery(kdtree, centers, d);
    assignments = zeros(class_num, size(d,2));
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(d, centers, assignments);
    siftvec_mb(m, :) = enc;
end
for m = 1:num_mm
    [f, d] = vl_sift(img_mm{m});
    d = im2double(d);
    nn = vl_kdtreequery(kdtree, centers, d);
    assignments = zeros(class_num, size(d,2));
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(d, centers, assignments);
    siftvec_mm(m, :) = enc;
end
% =================================================================

vec = [siftvec_mb; siftvec_mm];