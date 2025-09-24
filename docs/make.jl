using Documenter
using NowcastAutoGP

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
    pages = [
        "Home" => "index.md",
        "Getting started" => "vignettes/tutorial.md",
        "API Reference" => "api.md"
    ],
    clean = true,
    checkdocs = :exports
)

deploydocs(
    repo = "github.com/CDCgov/NowcastAutoGP.git",
    devbranch = "main"
)
