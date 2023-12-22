using LazySets: center

function propagate(X, Z, net; k=4, Y=BallInf(0.5 * ones(dim_out(net.layers[1])), 0.5))
    algo = SimpleBidirectional(BoxForward(BoxAffineMap()), BoxBackward())
    Xs = []
    Ys = []
    Zs = []
    push!(Xs, X)
    push!(Ys, Y)
    push!(Zs, Z)

    for i in 1:k
        Xi, Yi, Zi = bidirectional(X, Z, net, algo; get_intermediate_results=true)
        push!(Xs, Xi)
        push!(Ys, Yi)
        push!(Zs, Zi)
        X = Xi
        Z = Zi
        if isempty(X) || isempty(Z)
            break
        end
    end

    return (Xs, Ys, Zs)
end

function extract_bounds(Xs, Ys, Zs)
    ε = 0.01
    b1 = [isempty(X) ? (k > 1 ? (center(Xs[k - 1], 1) + ε, center(Xs[k - 1], 1) - ε) : (0.5, 0.5)) :
          (low(X, 1), high(X, 1)) for (k, X) in enumerate(Xs)]
    b2 = [isempty(X) ? (k > 1 ? (center(Xs[k - 1], 2) + ε, center(Xs[k - 1], 2) - ε) : (0.5, 0.5)) :
          (low(X, 2), high(X, 2)) for (k, X) in enumerate(Xs)]
    b3 = [isempty(Y) ? (k > 1 ? (center(Ys[k - 1], 1) + ε, center(Ys[k - 1], 1) - ε) : (0.5, 0.5)) :
          (low(Y, 1), high(Y, 1)) for (k, Y) in enumerate(Ys)]
    b4 = [isempty(Y) ? (k > 1 ? (center(Ys[k - 1], 2) + ε, center(Ys[k - 1], 2) - ε) : (0.5, 0.5)) :
          (low(Y, 2), high(Y, 2)) for (k, Y) in enumerate(Ys)]
    b5 = [isempty(Z) ? (k > 1 ? (center(Zs[k - 1], 1) + ε, center(Zs[k - 1], 1) - ε) : (0.5, 0.5)) :
          (low(Z, 1), high(Z, 1)) for (k, Z) in enumerate(Zs)]
    return b1, b2, b3, b4, b5
end

# neural network
W1 = [-4.60248 4.74295;
      -3.19378 2.90011]
b1 = [2.74108,
      -1.49695]
l1 = DenseLayerOp(W1, b1, Sigmoid())
W2 = [-4.57199 4.64925;]
b2 = [2.10176]
l2 = DenseLayerOp(W2, b2, Sigmoid())
net = FeedforwardNetwork([l1, l2])

# low-high domain
dom1 = cartesian_product(Interval(0.0, 0.2), Interval(0.8, 1.0))
codom1 = Interval(0.0, 1.0)
Xs1, Ys1, Zs1 = propagate(dom1, codom1, net)
bounds1 = extract_bounds(Xs1, Ys1, Zs1)

# low-low domain
dom2 = cartesian_product(Interval(0.0, 0.2), Interval(0.0, 0.2))
codom2 = Interval(0.0, 1.0)
Xs2, Ys2, Zs2 = propagate(dom2, codom2, net)
bounds2 = extract_bounds(Xs2, Ys2, Zs2)

# high-full domain + high codomain
dom3 = cartesian_product(Interval(0.8, 1.0), Interval(0.0, 1.0))
codom3 = Interval(0.5, 1.0)
Xs3, Ys3, Zs3 = propagate(dom3, codom3, net)
bounds3 = extract_bounds(Xs3, Ys3, Zs3)

# high-full domain + low codomain
dom4 = cartesian_product(Interval(0.8, 1.0), Interval(0.0, 1.0))
codom4 = Interval(0.0, 0.5)
Xs4, Ys4, Zs4 = propagate(dom4, codom4, net)
bounds4 = extract_bounds(Xs4, Ys4, Zs4)

# high-high domain + high codomain
dom5 = cartesian_product(Interval(0.8, 1.0), Interval(0.8, 1.0))
codom5 = Interval(0.5, 1.0)
Xs5, Ys5, Zs5 = propagate(dom5, codom5, net)
bounds5 = extract_bounds(Xs5, Ys5, Zs5)
