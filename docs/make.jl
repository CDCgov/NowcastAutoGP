using Documenter
using Literate
using NowcastAutoGP

pages = ["Home" => "index.md"]
# Generate Documenter-flavored markdown from Literate.jl scripts
# Add them to the `pages` list to include them in the documentation navigation
include("make_vignettes.jl")

# Generate API reference documentation from docstrings in the source code
push!(pages, "API Reference" => "api.md")

makedocs(
    sitename = "NowcastAutoGP.jl",
    authors = "Centers for Disease Control and Prevention",
    modules = [NowcastAutoGP],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://cdcgov.github.io/NowcastAutoGP/",
        assets = ["assets/material-theme.css", "assets/material-theme.js"],
        mathengine = Documenter.MathJax3()
    ),
    pages = pages,
    clean = true,
    checkdocs = :exports
)

deploydocs(
    repo = "github.com/CDCgov/NowcastAutoGP.git",
    devbranch = "main"
)
