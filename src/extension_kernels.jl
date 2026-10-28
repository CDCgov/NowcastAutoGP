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
also be uncertain. When `covariance_kernels` reports the fitted kernel in the original time units,
only the amplitude is rescaled (by the time-transform slope); the zero anchor is retained.

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

# `reparameterize` re-expresses the fitted amplitude in the original time units. AutoGP only
# calls it to report kernels (via `covariance_kernels`), never on the fit/predict path, which
# works entirely in normalized `[0, 1]` time. Under an input warp f(t) = slope*t + intercept,
# min(f(t), f(u)) = slope*min(t, u) + intercept, so exact equivalence would need an added
# `intercept * amplitude` constant. A `RandomWalk` is pinned to zero variance at its anchor and
# cannot carry that constant, so we scale the amplitude by `slope` and keep the anchor at
# normalized zero. This is exact only for pure scalings (intercept == 0); under AutoGP's `[0, 1]`
# time transform it reports the slope-scaled amplitude anchored at the start of the data.
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
level and slope should also be uncertain. When `covariance_kernels` reports the fitted kernel in
the original time units, only the amplitude is rescaled (by the time-transform slope); the zero
anchor is retained.

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

# `reparameterize` re-expresses the fitted amplitude in the original time units. AutoGP only
# calls it to report kernels (via `covariance_kernels`), never on the fit/predict path, which
# works entirely in normalized `[0, 1]` time. A pure input scaling t -> slope*t contributes three
# powers of slope to the covariance. A nonzero intercept would additionally spill lower-order
# terms that a zero-anchored integrated Brownian motion cannot represent, so we scale the
# amplitude by `slope^3` and keep the anchor at normalized zero. This is exact only for pure
# scalings (intercept == 0); under AutoGP's `[0, 1]` time transform it reports the slope-scaled
# amplitude anchored at the start of the data.
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
