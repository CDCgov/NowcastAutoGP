using Documenter
using NowcastAutoGP

# Doc pages
pages = [
    "Home" => "index.md",
    "Getting Started" => "vignettes/getting-started.md",
    "API Reference" => "api.md",
]

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
