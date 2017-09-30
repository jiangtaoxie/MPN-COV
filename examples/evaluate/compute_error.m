function error_state = compute_error(x,label)
%COMPUTE_ERROR 
%  Created by Jiangtao Xie    
error_state = [0 0 0 0];
predictions = x;
predictions(:,:,:,1) = mean(predictions,4);
[~,predictions] = sort(predictions, 3, 'descend');
if predictions(1) == label
    error_state(1) = 1;
end
top5 = predictions(1:5);
if ~isempty(find(top5 == label))
    error_state(2) = 1;
end
xmax = max(x,[],3) ;
ex = exp(bsxfun(@minus, x, xmax));
ex(:,:,:,1) = mean(ex,4);
[~,ex] = sort(ex, 3, 'descend');
if ex(1) == label
    error_state(3) = 1;
end
top5 = ex(1:5);
if ~isempty(find(top5 == label))
    error_state(4) = 1;
end
end

