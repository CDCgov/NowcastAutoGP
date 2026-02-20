using Literate
vignettes_to_make = ["getting-started.jl"]

# Generate markdown with executed output from Literate.jl scripts.
# Uses CommonMarkFlavor so Documenter.jl won't re-evaluate code blocks.
# Run this locally when vignettes change; commit the generated .md and images.
for vignettefile in vignettes_to_make
    Literate.markdown(
        joinpath(@__DIR__, "vignettes", vignettefile),
        joinpath(@__DIR__, "src", "vignettes");
        execute = true,
        flavor = Literate.CommonMarkFlavor(),
        mdstrings = true,
        credit = true
    )
end
