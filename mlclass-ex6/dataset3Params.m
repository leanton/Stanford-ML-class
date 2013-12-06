function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Algorithm is to look over all suggested values of C and sigma
% Than train each example and write error on validation set values to matrix error_val
% Than find indeces of smallest element in this matrix
suggested_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
error_val  = zeros(length(suggested_values), length(suggested_values));
for i=1:length(suggested_values)
	for j=1:length(suggested_values)
		C_var = suggested_values(i);
		sigma_var = suggested_values(j);
		model = svmTrain(X, y, C_var, @(x1, x2) gaussianKernel(x1, x2, sigma_var));
		predictions = svmPredict(model, Xval);
		error_val(i, j) = mean(double(predictions ~= yval));
	end
end

[minval,ind] = min(error_val(:));
[I,J] = ind2sub([size(error_val,1) size(error_val,2)],ind);
C = suggested_values(I);
sigma = suggested_values(J);

% =========================================================================

end
