using .TaylorIntegration: @taylorize
import .TaylorIntegration: TaylorIntegration, Taylor1, TaylorN  # `@taylorize` assumes that these symbols are defined

# Verisig helper functions (need to be defined in this separate file because of
# the macro)

# Eq. (4)-(6)
# d(σ(x))/dx = σ(x) * (1 - σ(x))
# g(t, x) = σ(tx) = 1 / (1 + exp(-tx))
# dg(t, x)/dt = g'(t, x) = x * g(t, x) * (1 - g(t, x))
@taylorize function _Verisig_sigmoid!(dx, x, p, t)
    n = div(length(x), 2)
    for i in 1:n
        xᴶ = x[i]
        xᴾ = x[n + i]
        dx[i] = zero(xᴶ)
        dx[n + i] = xᴶ * (xᴾ - xᴾ^2)
    end
    return dx
end

# Footnote 3
# d(tanh(x))/dx = 1 - tanh(x)^2
# g(t, x) = tanh(tx)
# dg(t, x)/dt = g'(t, x) = x * (1 - g(t, x)^2)
@taylorize function _Verisig_tanh!(dx, x, p, t)
    n = div(length(x), 2)
    for i in 1:n
        xᴶ = x[i]
        xᴾ = x[n + i]
        dx[i] = zero(xᴶ)
        dx[n + i] = xᴶ * (1 - xᴾ^2)
    end
    return dx
end

eval(load_Verisig())
# COV_EXCL_STOP
