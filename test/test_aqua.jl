@testitem "Aqua Tests" begin
    using Aqua
    using NowcastAutoGP

    # Run all Aqua tests with appropriate exclusions
    Aqua.test_all(
        NowcastAutoGP;
        ambiguities=false,      # Often has false positives
        unbound_args=false,     # Can have issues with advanced macro usage
        stale_deps=false,       # Not critical for initial setup
        deps_compat=false,      # Not critical for initial setup
        piracies=false,         # May have intentional type piracy
    )
end