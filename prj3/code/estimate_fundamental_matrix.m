% Fundamental Matrix Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
[n, m] = size(Points_a);
A = zeros(n, 9);
for i = 1: n
   current_a = Points_a(i, :);
   current_b = Points_b(i, :);
   u_a = current_b(1, 1);
   u_b = current_a(1, 1);
   v_a = current_b(1, 2);
   v_b = current_a(1, 2);
   A(i, :) = [u_a*u_b u_a*v_b u_a v_a*u_b v_a*v_b v_a u_b v_b 1];
end

[U, S, V] = svd(A);
F = V(:, end);
F_matrix = reshape(F, [3 3])';

[U, S, V] = svd(F_matrix);
S(3, 3) = 0;
F_matrix = U * S * V';
%This is an intentionally incorrect Fundamental matrix placeholder
% F_matrix = [0  0     -.0004; ...
%             0  0      .0032; ...
%             0 -0.0044 .1034];
%         
end

