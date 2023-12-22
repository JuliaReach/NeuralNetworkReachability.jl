using NeuralNetworkReachability.BackwardAlgorithms: _linear_map_inverse

W1 = [0.30 0.53
      0.77 0.42]
b1 = [0.43
      -0.42]
l1 = DenseLayerOp(W1, b1, LeakyReLU(0.01))
W2 = [0.17 -0.07
      0.71 -0.06]
b2 = [-0.01
      0.49]
l2 = DenseLayerOp(W2, b2, LeakyReLU(0.02))
W3 = [0.35 0.17
      -0.04 0.08]
b3 = [0.03
      0.17]
l3 = DenseLayerOp(W3, b3, Id())
layers = [l1, l2, l3]
net = FeedforwardNetwork(layers)

dom = BallInf([0.5, 0.5], 0.5)

Y = forward(dom, net, BoxForward(BoxAffineMap()))

# box approximation
algo = BoxBackward()
X3a = box_approximation(backward(Y, l3, algo))
X2a = box_approximation(backward(X3a, l2, algo))
X1a = backward(X2a, l1, algo)

# polyhedral computation
quadrants = [HPolyhedron([HalfSpace([-1.0, 0], 0.0), HalfSpace([0, -1.0], 0.0)]),
             HPolyhedron([HalfSpace([-1.0, 0], 0.0), HalfSpace([0, 1.0], 0.0)]),
             HPolyhedron([HalfSpace([1.0, 0], 0.0), HalfSpace([0, -1.0], 0.0)]),
             HPolyhedron([HalfSpace([1.0, 0], 0.0), HalfSpace([0, 1.0], 0.0)])]

# manual implementation
algo = PolyhedraBackward()
X3b = backward(Y, l3, algo)
X2P = filter!(!isempty, [intersection(X3b, Q) for Q in quadrants])
X2A = UnionSetArray([X2P[1],  # _linear_map_inverse([1.0, 1.0], X2P[1]),  # identity
                     _linear_map_inverse([l2.activation.slope, 1.0], X2P[2])])
X2b = backward(X2A, l2.weights, l2.bias, algo)
X1P = filter!(!isempty, array(flatten(UnionSetArray([intersection(X2b, Q) for Q in quadrants]))))
X1A = UnionSetArray([X1P[1],  # _linear_map_inverse([1.0, 1.0], X1P[1]),  # identity
                     X1P[2],  # _linear_map_inverse([1.0, 1.0], X1P[2]),  # identity
                     _linear_map_inverse([1.0, l1.activation.slope], X1P[3]),
                     _linear_map_inverse([l1.activation.slope, l1.activation.slope], X1P[4])])
X1b = backward(X1A, l1.weights, l1.bias, algo)

# algorithm
algo = PolyhedraBackward()
X1c = backward(Y, net, algo)

@test isequivalent(X1b, X1c)
