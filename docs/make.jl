using TheEggman
using Documenter

DocMeta.setdocmeta!(TheEggman, :DocTestSetup, :(using TheEggman); recursive=true)

makedocs(;
    modules=[TheEggman],
    authors="Jacob Gunnell <jacob.r.gunnell@gmail.com> and contributors",
    sitename="TheEggman.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
