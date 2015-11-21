% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete. 20 random points
x = [];
y = [];
[h, w, layers] = size(image);
sigma = 2;
filter = fspecial('Gaussian', 5, 3);


xderivative_filter = imfilter(filter, [1,-1], 'symmetric');
yderivative_filter = imfilter(filter, [1;-1], 'symmetric');
h_derivative = zeros(h,w,layers);
v_derivative = zeros(h,w,layers);
ix2 = zeros(h,w,layers);
iy2 = zeros(h,w,layers);
ixiy = zeros(h,w,layers);

sigma = 2;
w = fspecial('Gaussian', 3, sigma);
for l = 1:layers
    h_derivative(:,:,l) = imfilter(image(:,:,l), xderivative_filter);
    v_derivative(:,:,l) = imfilter(image(:,:,l), yderivative_filter);

    ix2(:,:,l) = imfilter(h_derivative(:,:,l).*h_derivative(:,:,l), w);
    iy2(:,:,l) = imfilter(v_derivative(:,:,l).*v_derivative(:,:,l), w);
    ixiy(:,:,l) = imfilter(h_derivative(:,:,l).*v_derivative(:,:,l), w);
    
end

alpha = 0.06;
threshold = 0.01;

har = ix2.*iy2-ixiy.*ixiy-alpha*(ix2+iy2).*(ix2+iy2);


har(1:feature_width, :, :) = 0;
har(end-feature_width:end, :, :) = 0;
har(:, 1:feature_width, :) = 0;
har(:, end-feature_width:end, :) = 0;
maxval = max(max(har));
corner = har.*(har > threshold*maxval);
result = colfilt(corner, [3 3], 'sliding', @max);
corner = corner.*(corner==result);
[yl, xl]=find(corner);
    
x = [x;xl];
y = [y;yl];

% [har, layer] = sort(har, 3,'descend');
% maxval = max(max(har(:,:,1)));
% corner = har(:,:,1).*(har(:,:,1) > threshold*maxval);
% %l = layer.*(har[:,:,1]>threshold*maxval);
% %corner = har.*(har>threshold*maxval);
% result = colfilt(corner, [3 3], 'sliding', @max);
% corner = corner.*(corner==result);
% [yl, xl]=find(corner);
% zl = layer(yl, xl, 1);
% y = [y ; yl];
% x = [x ; xl];
% z = [z ; zl];
% maxval = max(max(har));
% cornerness = har.*(har>threshold * maxval);
% 
% result = colfilt(cornerness, [3 3], 'sliding', @max);
% cornerness = cornerness.*(cornerness == result);
% 
% [y, x] = find(cornerness);


end

