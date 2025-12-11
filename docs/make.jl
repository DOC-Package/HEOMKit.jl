using KaisouEOM
using Documenter

DocMeta.setdocmeta!(KaisouEOM, :DocTestSetup, :(using KaisouEOM); recursive=true)

makedocs(;
    modules=[KaisouEOM],
    authors="Hideaki Takahashi",
    sitename="KaisouEOM.jl",
    format=Documenter.HTML(;
        canonical="https://htkhsh.github.io/KaisouEOM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Bath and Noise" => "manual/bath.md",
            "HEOM System" => "manual/heom.md",
            "Time Evolution" => "manual/evolution.md",
            "Constants" => "manual/constants.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/htkhsh/KaisouEOM.jl",
    devbranch="main",
)
