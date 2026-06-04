@testitem "make_and_fit_model forwards config to GPModel" setup = [ModelFittingData] begin
    data = create_transformed_data(dates, values; transformation = identity)

    cfg = GPConfig()
    model = make_and_fit_model(data; config = cfg, test_params...)

    @test model isa AutoGP.GPModel
    # config is stored by reference on the GPModel, so the exact object should round-trip
    @test model.config === cfg
end

@testitem "make_and_fit_model forwards a customised kernel-structure config" setup = [ModelFittingData] begin
    data = create_transformed_data(dates, values; transformation = identity)

    cfg = GPConfig(node_dist_leaf = [0.0, 0.25, 0.25, 0.25, 0.25], changepoints = false)
    model = make_and_fit_model(data; config = cfg, test_params...)

    @test model.config.node_dist_leaf == [0.0, 0.25, 0.25, 0.25, 0.25]
    @test model.config.changepoints == false
end

@testitem "make_and_fit_model forwards a custom prior" setup = [ModelFittingData] begin
    data = create_transformed_data(dates, values; transformation = identity)

    # copy-and-update a single nested prior entry (the documented dependency-free pattern)
    prior = deepcopy(GPConfig().prior)
    prior[:period][:mu] = log(1.0)
    cfg = GPConfig(prior = prior)
    model = make_and_fit_model(data; config = cfg, test_params...)

    @test model.config.prior[:period][:mu] == log(1.0)
    # untouched sibling entries keep their defaults
    @test model.config.prior[:period][:sigma] == GPConfig().prior[:period][:sigma]
    @test model.config.prior[:gamma] == GPConfig().prior[:gamma]
end

@testitem "make_and_fit_model requires n_mcmc/n_hmc (pure passthrough)" setup = [ModelFittingData] begin
    data = create_transformed_data(dates, values; transformation = identity)

    # n_mcmc / n_hmc are now required pass-through kwargs of AutoGP.fit_smc!,
    # so omitting them surfaces as an UndefKeywordError rather than a wrapper default.
    @test_throws UndefKeywordError make_and_fit_model(data; n_particles = 1)
end
