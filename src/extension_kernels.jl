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

##########################################
### IntegratedBrownianMotion kernel ######
##########################################

@doc raw"""
    IntegratedBrownianMotion(origin[, amplitude=1])

Once-integrated Brownian motion (integrated Wiener process) covariance kernel.

```math
k(t, t') = \theta_2 \, \frac{a^2 (3b - a)}{6},
\qquad a = \min(t, t') - \theta_1, \quad b = \max(t, t') - \theta_1
```

This is the covariance of ``X(t) = \int_{\theta_1}^{t} W(s)\,\mathrm{d}s``, the integral of a
Brownian motion that starts at the `origin` ``\theta_1`` (where both the value and the
variance are zero); `amplitude` ``\theta_2`` scales the variance. Draws are smoother
(once-differentiable) than [`RandomWalk`](@ref) draws — an integrated-random-walk prior, a
natural choice for trends whose *rate of change* drifts like a random walk.

The kernel is positive-definite only when `origin <= min(t)` over the evaluated time points,
so that every `a, b >= 0`.

Like [`RandomWalk`](@ref), this kernel is defined in `NowcastAutoGP` (not `AutoGP`) by
extending `AutoGP`'s `GP` interface (`eval_cov`, `reparameterize`, `rescale`).

### Connection to cubic splines

The predictive mean of a GP with kernel composition of primitives `Const` + `Linear` + `IntegratedBrownianMotion`
is a cubic spline (piecewise cubic polynomial) between data points and linear outside the data range, see [Rasmussen & Williams, 2006, §6.3].
This means that a GP with this kernel composition represents a Bayesian cubic spline model, with the `IntegratedBrownianMotion` kernel controlling the smoothness of the spline.
Note that the kernel covariance here seems to differ in form from the one in Rasmussen & Williams, but noting that |t - t'| = b - a, the two forms are equivalent.
"""
struct IntegratedBrownianMotion <: LeafNode
    origin::Real
    amplitude::Real
    IntegratedBrownianMotion(origin::Real, amplitude::Real = 1) = new(origin, amplitude)
end

function eval_cov(node::IntegratedBrownianMotion, t1, t2)
    a = min(t1, t2) - node.origin
    b = max(t1, t2) - node.origin
    return node.amplitude * a^2 * (3b - a) / 6
end

function eval_cov(node::IntegratedBrownianMotion, ts::Vector{Float64})
    a = min.(ts, ts') .- node.origin
    b = max.(ts, ts') .- node.origin
    return node.amplitude .* a .^ 2 .* (3 .* b .- a) ./ 6
end

# Input transform f(t) = a*t + b (a = t.slope > 0, b = t.intercept). The covariance is a
# degree-3 homogeneous function of (t - origin, t' - origin) (a double integral of min, which
# is degree-1), so under f the origin transforms as for `RandomWalk` while the amplitude
# absorbs a factor of slope^3:
#   origin'    = (origin - b) / a
#   amplitude' = a^3 * amplitude
function reparameterize(node::IntegratedBrownianMotion, t::LinearTransform)
    origin = (node.origin - t.intercept) / t.slope
    amplitude = t.slope^3 * node.amplitude
    return IntegratedBrownianMotion(origin, amplitude)
end

# Output transform Y = a*X + b scales variance by a^2; the origin is unchanged.
function rescale(node::IntegratedBrownianMotion, t::LinearTransform)
    return IntegratedBrownianMotion(node.origin, t.slope^2 * node.amplitude)
end

function pretty(node::IntegratedBrownianMotion)
    return @sprintf("IBM(%1.2f; %1.2f)", node.origin, node.amplitude)
end
