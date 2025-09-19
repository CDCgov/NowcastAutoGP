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
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    # repo = "https://github.com/CDCgov/NowcastAutoGP/",
    clean = true,
    checkdocs = :exports,
    remotes = nothing
)

deploydocs(
    repo = "github.com/CDCgov/NowcastAutoGP.git",
    devbranch = "main",
)
