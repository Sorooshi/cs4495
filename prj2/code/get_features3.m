% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, z, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

% Placeholder that you can delete. Empty features.
features = zeros(size(x,1), 128);

cutoff_frequency = 2;
small_guassian = fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency);
image = imfilter(image, small_guassian, 'symmetric');

smooth_filter = fspecial('Gaussian', 2*4+1, 4);
[height, width, layers] = size(image);
all_gradient = zeros(height, width, layers, 8);
xd = fspecial('sobel')';
% gx = imfilter(imfilter(image, xd), smooth_filter);
% gy = imfilter(imfilter(image, xd'), smooth_filter);
for l = 1:layers
    
all_gradient(:,:,l,1) = imfilter(imfilter(image(:,:,l), xd,'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,2) = imfilter(imfilter(image(:,:,l), imrotate(xd,45,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,3) = imfilter(imfilter(image(:,:,l), imrotate(xd,90,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,4) = imfilter(imfilter(image(:,:,l), imrotate(xd,135,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,5) = imfilter(imfilter(image(:,:,l), imrotate(xd,180,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,6) = imfilter(imfilter(image(:,:,l), imrotate(xd,225,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,7) = imfilter(imfilter(image(:,:,l), imrotate(xd,270,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
all_gradient(:,:,l,8) = imfilter(imfilter(image(:,:,l), imrotate(xd,315,'bilinear'), 'symmetric'), smooth_filter, 'symmetric');
end

%weight_filter = fspecial('Gaussian', feature_width*2+1, feature_width/2);
for i=1:size(x,1)
    xi = x(i);
    yi = y(i);
    feature_vector = zeros(1,128);

        window = all_gradient(yi-8:yi+7, xi-8:xi+7, :);

        size(window);
        %window = imfilter(window, weight_filter, 'symmetric');
        for m = 0:3
            for n = 0:3
                patch = window(m*4+1:(m+1)*4, n*4+1:(n+1)*4,:);
                [sorted_gradient, gra_ind] = sort(patch, 3, 'descend');
                maxg = sorted_gradient(:,:,1:2);
                indg = gra_ind(:,:,1:2);
                gx = abs(patch(:,:,1));
                gy = abs(patch(:,:,3));
                
                maxg(:,:,1) = maxg(:,:,1)-maxg(:,:,2);
                maxg(:,:,2) = maxg(:,:,2).*sqrt(2);
               
                for j = 1:8
                    gsum = sum(maxg(indg==j));
                    feature_vector(1, 32*m+8*n+j) = feature_vector(1, 32*m+8*n+j)+gsum;
                    gsum = gsum*0.5;
                    if m-1>=0
                        feature_vector(1, 32*(m-1)+8*n+j) = feature_vector(1, 32*(m-1)+8*n+j)+gsum;
                    end
                    
                    if m+1 <= 3
                        feature_vector(1, 32*(m+1)+8*n+j) = feature_vector(1, 32*(m+1)+8*n+j)+gsum;
                    end
                    
                    if n-1 >= 0
                        feature_vector(1, 32*m+8*(n-1)+j) = feature_vector(1, 32*m+8*(n-1)+j)+gsum;
                    end
                    
                    if n+1 <= 3
                        feature_vector(1, 32*m+8*(n+1)+j) = feature_vector(1, 32*m+8*(n+1)+j)+gsum;
                    end
                end
                
                
            end
        end
    
    norm_factor = 1/norm(feature_vector,1);
    feature_vector = feature_vector*norm_factor;
    maxval = max(feature_vector);
    threshold = 0.05*maxval;
    feature_vector(feature_vector > threshold) = threshold;

    norm_factor = 1/norm(feature_vector,1);
    feature_vector = feature_vector*norm_factor;
    size(feature_vector);
    size(features);
    features(i,:) = feature_vector;
end

    
end








