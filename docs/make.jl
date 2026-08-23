using HEOMKit
using Documenter

DocMeta.setdocmeta!(HEOMKit, :DocTestSetup, :(using HEOMKit); recursive=true)

makedocs(;
    modules=[HEOMKit],
    authors="Hideaki Takahashi",
    sitename="HEOMKit.jl",
    format=Documenter.HTML(;
        canonical="https://htkhsh.github.io/HEOMKit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Bath and Noise" => "manual/bath.md",
            "HEOM System" => "manual/heom.md",
            "Time Evolution" => "manual/evolution.md",
            "Performance" => "manual/performance.md",
            "Constants" => "manual/constants.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/htkhsh/HEOMKit.jl",
    devbranch="main",
)
