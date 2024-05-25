function example_network_222(act::ActivationFunction=ReLU())
    return FeedforwardNetwork([DenseLayerOp([1.0 2.0; -1.0 -2.0], [1.0, -1.0], act),
                               DenseLayerOp([-1.0 -2.0; 1.0 2.0], [-1.0, 1.0], Id())])
end

function example_network_232(act::ActivationFunction=ReLU())
    return FeedforwardNetwork([DenseLayerOp([1.0 2.0; -1.0 -2.0; 3.0 -3.0], [1.0, -1.0, 1.0], act),
                               DenseLayerOp([-1.0 -2.0 -3.0; 1.0 2.0 3.0], [-1.0, 1.0], Id())])
end

function example_network_1221(act::ActivationFunction=ReLU())
    return FeedforwardNetwork([DenseLayerOp(hcat([1.0; 1.0]), [1.5, 1.5], act),
                               DenseLayerOp([2.0 2.0; 2.0 2.0], [2.5, 2.5], act),
                               DenseLayerOp([3.0 3.0;], [3.5], Id())])
end

function example_network_AI2()
    return FeedforwardNetwork([DenseLayerOp([2.0 -1.0; 0.0 1.0], [0.0, 0.0], ReLU())])
end
