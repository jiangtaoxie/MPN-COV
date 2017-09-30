function score = predict_dagnn(data,label,net)
%PREDICT_DAGNN  Predict the input image for DAGNN network
% Created by Jiangtao Xie
inputs = {'input', data, 'label', label} ;
net.eval(inputs);
prediction_id = net.getVarIndex('prediction');
score = net.vars(prediction_id).value;
end

