vignettes_to_make = [
    "Getting started" => "tutorial.jl",
]

# Generate Documenter-flavored markdown from Literate.jl scripts
for vignette_pair in vignettes_to_make
    vignettefile = vignette_pair.second
    Literate.markdown(
        joinpath(@__DIR__, "vignettes", vignettefile),
        joinpath(@__DIR__, "src", "vignettes");
        eval = false,
        flavor = Literate.DocumenterFlavor(),
        mdstrings = true,
        credit = true
    )
    push!(pages, vignette_pair.first => "vignettes/$(replace(vignettefile, ".jl" => ".md"))")
end
