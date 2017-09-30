classdef MPN_COV_Pool_C < dagnn.Layer
  properties
    method = 'covariance_small'
    regu_method = 'power'
    alpha = 0.5
    epsilon = 1e-12
    opts = {'cuDNN'}
  end

  methods
      
    %% forward  function
    function outputs = forward(obj, inputs, params)
        % function upper = nn_forward_cov_pool(opts, lower, upper, masks)
        % res(i+1) = l.forward(l, res(i), res(i+1)) ;
        [outputs{1},outputs{2},outputs{3},outputs{4}] = vl_nnmpn_cov(inputs{1}, ...
                                  'epsilon',obj.epsilon, ...
                                  'alpha',obj.alpha, ...
                                  'method',obj.method, ...
                                  'regu_method',obj.regu_method, ...
                                   obj.opts{:});
    end
    

%      %% backwardAdvanced is modified slightly for custom layer
    function backwardAdvanced(obj, layer)
    %BACKWARDADVANCED Advanced driver for backward computation
    %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the backward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;
      derOutputs = {net.vars(out).der} ;  
      outputs = {net.vars(out).value};  
      if isempty(derOutputs{1}), return; end

      if net.conserveMemory
        % clear output variables (value and derivative)
        % unless precious
        for i = out
          if net.vars(i).precious, continue ; end
          net.vars(i).der = [] ;
          net.vars(i).value = [] ;
        end
      end

      % compute derivatives of inputs and paramerters
      [derInputs, derParams] = obj.backward(inputs , derOutputs, outputs) ;
      if ~iscell(derInputs) || numel(derInputs) ~= numel(in)
        error('Invalid derivatives returned by layer "%s".', layer.name);
      end

      % accumuate derivatives
      for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
          net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end

      for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
              || isempty(net.params(p).der)
          net.params(p).der = derParams{i} ;
        else
          net.params(p).der = vl_taccum(...
            1, net.params(p).der, ...
            1, derParams{i}) ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
        if net.numPendingParamRefs(p) == net.params(p).fanout
          if ~isempty(net.parameterServer) && ~net.holdOn
            net.parameterServer.pushWithIndex(p, net.params(p).der) ;
            net.params(p).der = [] ;
          end
        end
      end
    end
   
    %% backward function
    function [derInputs, derParams] = backward(obj, inputs, derOutputs,outputs)
        
       derInputs{1} = vl_nnmpn_cov(inputs{1}, ...
                                   outputs{2},... % aux_S
                                   outputs{3},... % aux_V
                                   outputs{4},... % aux_D
                                   derOutputs{1}, ...
                                  'epsilon',obj.epsilon, ...
                                  'alpha',obj.alpha, ...
                                  'method',obj.method, ...
                                  'regu_method',obj.regu_method, ...
                                   obj.opts{:});
       derParams  = {} ;
       
    end
  

    %% constructor
    function obj = MPN_COV_Pool_C(varargin)
      obj.load(varargin) ;
    end
  end
end
