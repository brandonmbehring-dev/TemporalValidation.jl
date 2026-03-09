using TemporalValidation
using Documenter

DocMeta.setdocmeta!(TemporalValidation, :DocTestSetup, :(using TemporalValidation); recursive=true)

makedocs(;
    modules=[TemporalValidation],
    warnonly=[:missing_docs],
    authors="Brandon Behring",
    sitename="TemporalValidation.jl",
    format=Documenter.HTML(;
        canonical="https://bbehring.github.io/TemporalValidation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bbehring/TemporalValidation.jl",
    devbranch="main",
)
