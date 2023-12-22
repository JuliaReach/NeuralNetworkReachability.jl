actual(x) = x^2 / 20
dom = Interval(-20.0, 20.0)
codom = Interval(0.0, 20.0)

net = read_POLAR(joinpath(@__DIR__, "parabola.network"))
algo = PolyhedraBackward()
N = 500
seed = 0
s = sample(dom, N; seed=seed)
x_train = hcat(sort!(Float32.(getindex.(s, 1)))...)
y_train = actual.(x_train)
arr_train = vec([Singleton([xi, yi]) for (xi, yi) in zip(x_train, y_train)])
x_net = [[x] for k in 1:N for x in x_train[1, k]]  # requires different type
y_net = net.(x_net)
arr_net = vec([Singleton([xi[1], yi[1]]) for (xi, yi) in zip(x_net, y_net)])

for k in 1:20
    local Y = Interval(k - 1, k)
    local preimage = backward(Y, net, algo)
end

# preimage of [100, 105]
Y = Interval(100, 105)
preimage = backward(Y, net, algo)
for (i, X) in enumerate(array(preimage))
    Xi = convert(Interval, X)
end

# preimage of [-1, 0]
Y = HalfSpace([1.0], 0.0)
preimage = backward(Y, net, algo)
