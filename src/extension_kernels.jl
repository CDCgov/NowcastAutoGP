import AutoGP.GP: LeafNode, eval_cov, reparameterize, rescale, pretty, LinearTransform

#############################
### RandomWalk kernel #######
#############################

@doc raw"""
    RandomWalk([amplitude=1])

Random walk (Wiener process) covariance kernel.

```math
k(t, t') = \theta_1 \min(t, t')
```

The process is anchored at time zero, where the variance is zero; `amplitude` ``\theta_1``
scales the per-unit-time variance. Draws from this kernel are continuous-time random walks
(Brownian motion): non-stationary, with variance growing linearly away from zero.

This kernel is positive semidefinite when evaluated on nonnegative time points. AutoGP rescales
time to `[0, 1]` before fitting, so anchoring the process at zero lets AutoGP sample only the
variance amplitude. Use a `Constant` kernel alongside `RandomWalk` when the initial level should
also be uncertain.

This kernel is defined in `NowcastAutoGP` (not `AutoGP`) by extending `AutoGP`'s `GP` interface
(`eval_cov`, `reparameterize`, `rescale`). It mirrors AutoGP's primitive kernel structure.
"""
struct RandomWalk <: LeafNode
    amplitude::Real
    RandomWalk(amplitude::Real = 1) = new(amplitude)
end

eval_cov(node::RandomWalk, t1, t2) = node.amplitude * min(t1, t2)

function eval_cov(node::RandomWalk, ts::Vector{Float64})
    return node.amplitude .* min.(ts, ts')
end

# Input scaling f(t) = a*t + b (a = t.slope > 0) contributes one power of slope to
# the random-walk covariance. The anchored representation does not carry a translated
# origin, so reparameterization keeps the zero anchor in the target coordinate system.
function reparameterize(node::RandomWalk, t::LinearTransform)
    amplitude = t.slope * node.amplitude
    return RandomWalk(amplitude)
end

# Output transform Y = a*X + b scales variance by a^2.
function rescale(node::RandomWalk, t::LinearTransform)
    return RandomWalk(t.slope^2 * node.amplitude)
end

pretty(node::RandomWalk) = @sprintf("RW(%1.2f)", node.amplitude)

##########################################
### IntegratedBrownianMotion kernel ######
##########################################

@doc raw"""
    IntegratedBrownianMotion([amplitude=1])

Once-integrated Brownian motion (integrated Wiener process) covariance kernel.

```math
k(t, t') = \theta_1 \, \frac{a^2 (3b - a)}{6},
\qquad a = \min(t, t'), \quad b = \max(t, t')
```

This is the covariance of ``X(t) = \int_0^{t} W(s)\,\mathrm{d}s``, the integral of a
Brownian motion that starts at zero (where both the value and the variance are zero);
`amplitude` ``\theta_1`` scales the variance. Draws are smoother (once-differentiable)
than [`RandomWalk`](@ref) draws -- an integrated-random-walk prior, a
natural choice for trends whose *rate of change* drifts like a random walk.

This kernel is positive semidefinite when evaluated on nonnegative time points. AutoGP rescales
time to `[0, 1]` before fitting, so anchoring the process at zero lets AutoGP sample only the
variance amplitude. Use `Constant` + `Linear` + `IntegratedBrownianMotion` when the initial
level and slope should also be uncertain.

Like [`RandomWalk`](@ref), this kernel is defined in `NowcastAutoGP` (not `AutoGP`) by
extending `AutoGP`'s `GP` interface (`eval_cov`, `reparameterize`, `rescale`).

### Connection to cubic splines

The predictive mean of a GP with kernel composition of primitives `Const` + `Linear` + `IntegratedBrownianMotion`
is a cubic spline (piecewise cubic polynomial) between data points and linear outside the data range, see [Rasmussen & Williams, 2006, §6.3].
This means that a GP with this kernel composition represents a Bayesian cubic spline model, with the `IntegratedBrownianMotion` kernel controlling the smoothness of the spline.
Note that the kernel covariance here seems to differ in form from the one in Rasmussen & Williams, but noting that |t - t'| = b - a, the two forms are equivalent.
"""
struct IntegratedBrownianMotion <: LeafNode
    amplitude::Real
    IntegratedBrownianMotion(amplitude::Real = 1) = new(amplitude)
end

function eval_cov(node::IntegratedBrownianMotion, t1, t2)
    a = min(t1, t2)
    b = max(t1, t2)
    return node.amplitude * a^2 * (3b - a) / 6
end

function eval_cov(node::IntegratedBrownianMotion, ts::Vector{Float64})
    a = min.(ts, ts')
    b = max.(ts, ts')
    return node.amplitude .* a .^ 2 .* (3 .* b .- a) ./ 6
end

# Input scaling f(t) = a*t + b (a = t.slope > 0) contributes three powers of slope to the
# integrated-Brownian-motion covariance. The anchored representation does not carry a
# translated origin, so reparameterization keeps the zero anchor in the target coordinate system.
function reparameterize(node::IntegratedBrownianMotion, t::LinearTransform)
    amplitude = t.slope^3 * node.amplitude
    return IntegratedBrownianMotion(amplitude)
end

# Output transform Y = a*X + b scales variance by a^2.
function rescale(node::IntegratedBrownianMotion, t::LinearTransform)
    return IntegratedBrownianMotion(t.slope^2 * node.amplitude)
end

function pretty(node::IntegratedBrownianMotion)
    return @sprintf("IBM(%1.2f)", node.amplitude)
end
