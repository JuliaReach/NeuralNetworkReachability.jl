#############
# Example 1 #
#############

W = [-0.46 0.32;]
b = [2.0]
Y = Interval(2.0, 3.0)
algo = PolyhedraBackward()

X = backward(Y, W, b, algo)

#############
# Example 3 #
#############

Y = X

X = backward(Y, ReLU(), algo)

@test length(X) == 3  # one set is empty
X₁ = X[1]
X₂ = X[2]
X₃ = ∅(2)
X₄ = X[3]

#############
# Example 4 #
#############

# neural network

W1 = [0.30 0.53
      0.77 0.42]
b1 = [0.43
      -0.42]
l1 = DenseLayerOp(W1, b1, ReLU())
W2 = [0.17 -0.07
      0.71 -0.06]
b2 = [-0.01
      0.49]
l2 = DenseLayerOp(W2, b2, ReLU())
W3 = [0.35 0.17
      -0.04 0.08]
b3 = [0.03
      0.17]
l3 = DenseLayerOp(W3, b3, Id())
layers = [l1, l2, l3]
net = FeedforwardNetwork(layers)

# samples

dom = BallInf([0.5, 0.5], 0.5)
seed = 100
x = sample(dom, 300; seed=seed)
y = net.(x)
x1 = [x[i] for i in eachindex(x) if y[i][1] >= y[i][2]]
x2 = [x[i] for i in eachindex(x) if y[i][1] < y[i][2]]
y1 = [yi for yi in y if yi[1] >= yi[2]]
y2 = [yi for yi in y if yi[1] < yi[2]]

# forward image under domain

Y = forward(dom, net, ConcreteForward())

# backward image under subset y2 >= y1

algo = PolyhedraBackward()
Y1 = HalfSpace([1.0, -1.0], 0.0)
Y2 = backward(Y1, l3, algo)
Y3 = backward(Y2, ReLU(), algo)
Y4 = backward(Y3, W2, b2, algo)
Y5 = backward(Y4, ReLU(), algo)
Y6 = backward(Y5, W1, b1, algo)
