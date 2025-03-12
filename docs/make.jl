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
    ],
)

deploydocs(;
    repo="github.com/htkhsh/KaisouEOM.jl",
    devbranch="main",
)
