import AutoGP.GP: LeafNode, eval_cov, reparameterize, rescale, pretty, LinearTransform

#############################
### RandomWalk kernel #######
#############################

@doc raw"""
    RandomWalk(origin[, amplitude=1])

Random walk (Wiener process) covariance kernel.

```math
k(t, t') = \theta_2 \left( \min(t, t') - \theta_1 \right)
```

The `origin` ``\theta_1`` is the time at which the process starts (where the variance is
zero); `amplitude` ``\theta_2`` scales the per-unit-time variance. Draws from this kernel are
continuous-time random walks (Brownian motion): non-stationary, with variance growing linearly
away from `origin`.

The kernel is positive-definite only when `origin <= min(t)` over the evaluated time points,
so that every `min(t, t') - origin >= 0`. This is the GP analogue of an i.i.d.-increment
random walk.

This kernel is defined in `NowcastAutoGP` (not `AutoGP`) by extending `AutoGP`'s `GP` interface
(`eval_cov`, `reparameterize`, `rescale`). It mirrors AutoGP's primitive kernel structure.
"""
struct RandomWalk <: LeafNode
    origin::Real
    amplitude::Real
    RandomWalk(origin::Real, amplitude::Real = 1) = new(origin, amplitude)
end

eval_cov(node::RandomWalk, t1, t2) = node.amplitude * (min(t1, t2) - node.origin)

function eval_cov(node::RandomWalk, ts::Vector{Float64})
    return node.amplitude .* (min.(ts, ts') .- node.origin)
end

# Input transform f(t) = a*t + b (a = t.slope > 0, b = t.intercept). We need
# k(f(t), f(u); θ) = k(t, u; θ'). Since min(a*t+b, a*u+b) = a*min(t,u) + b,
#   origin'    = (origin - b) / a       (identical to Linear's intercept rule)
#   amplitude' = a * amplitude          (degree-1 in t, so scales linearly with slope)
function reparameterize(node::RandomWalk, t::LinearTransform)
    origin = (node.origin - t.intercept) / t.slope
    amplitude = t.slope * node.amplitude
    return RandomWalk(origin, amplitude)
end

# Output transform Y = a*X + b scales variance by a^2; the origin is unchanged.
function rescale(node::RandomWalk, t::LinearTransform)
    return RandomWalk(node.origin, t.slope^2 * node.amplitude)
end

pretty(node::RandomWalk) = @sprintf("RW(%1.2f; %1.2f)", node.origin, node.amplitude)
