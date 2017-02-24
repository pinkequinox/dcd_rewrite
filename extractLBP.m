function [ lbpvec ] = extractLBP( img, cellnum, neighbor, r )
num = length(img);
for m = 1:num
    J = extractLBPFeatures(img{m}, 'Upright',false, 'CellSize', floor(size(img{m})/cellnum), 'NumNeighbors', neighbor, 'Radius', r);
    lbpvec(m, :) = J;
end


end

