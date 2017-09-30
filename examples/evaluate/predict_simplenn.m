function score = predict_simplenn(data,label,net)
%PREDICT_SIMPLENN Predict the input image for SIMPLENN network
%  Created by Jiangtao Xie
net.layers{end}.class = label;
res = vl_simplenn(net,data,[],[],'mode', 'test');
score = gather(res(end-1).x);
end

