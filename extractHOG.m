function [ hogvec ] = extractHOG( img, cellnum, blknum )

num = length(img);
for idx = 1:num
    hogvec(idx, :) = extractHOGFeatures(img{idx}, 'CellSize', floor(size(img{idx})/cellnum) ,'BlockSize',[blknum blknum], 'BlockOverlap', [0 0]);

end

