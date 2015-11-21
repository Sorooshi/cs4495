% RANSAC Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
% that you wrote for part II.

%placeholders, you can delete all of this
N = 0.8;
max_loop = 1000;

[n, m] = size(matches_a);
max_inlines = 0;
max_a = 0;
max_b = 0;
for i = 1 : max_loop
    sample_idx = randperm(n, 8);
    sample_a  = matches_a(sample_idx, :);
    sample_b = matches_b(sample_idx, :);
    current_f_matrix = estimate_fundamental_matrix(sample_a, sample_b);% this estimate the f_matrix for points in sample b
    [inlines_a, inlines_b] = get_inliner(matches_a, matches_b, current_f_matrix, n);
    [num, ~] = size(inlines_a);
    if num / n > N
        i = max_loop+1;
        max_a = inlines_a;
        max_b = inlines_b;
    elseif max_inlines < num
        max_inlines = num;
        max_a = inlines_a;
        max_b = inlines_b;
    end
end
inliers_a = max_a;
inliers_b = max_b;
Best_Fmatrix = estimate_fundamental_matrix(inlines_a, inlines_b);
end


function [inlines_a, inlines_b] = get_inliner(matches_a, matches_b, f_matrix, n)
       inlines_a = [];
       inlines_b = [];
       threshold = 0.05;
       for i = 1 : n
           if abs([matches_b(i, :) 1] * f_matrix * [matches_a(i, :) 1]') < threshold
               inlines_a = [inlines_a;matches_a(i, :)];
               inlines_b = [inlines_b;matches_b(i, :)];
           end
       end
end

