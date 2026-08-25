using TheEggman
using Documenter

DocMeta.setdocmeta!(TheEggman, :DocTestSetup, :(using TheEggman); recursive=true)

makedocs(;
    modules=[TheEggman],
    authors="Jacob Gunnell <jgunnell@umd.edu> and contributors",
    sitename="TheEggman.jl",
    format=Documenter.HTML(;
        canonical="https://QuantumSavory.github.io/TheEggman.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QuantumSavory/TheEggman.jl",
    devbranch="main",
)
