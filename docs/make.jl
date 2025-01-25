using Documenter, NeuralNetworkReachability, DocumenterCitations

DocMeta.setdocmeta!(NeuralNetworkReachability, :DocTestSetup,
                    :(using NeuralNetworkReachability); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:alpha)

makedocs(; sitename="NeuralNetworkReachability.jl",
         modules=[NeuralNetworkReachability],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
                                collapselevel=1,
                                assets=["assets/aligned.css", "assets/citations.css"]),
         pagesonly=true,
         plugins=[bib],
         pages=["Home" => "index.md",
                "Library" => Any["ForwardAlgorithms" => "lib/ForwardAlgorithms.md",
                                 "BackwardAlgorithms" => "lib/BackwardAlgorithms.md",
                                 "BidirectionalAlgorithms" => "lib/BidirectionalAlgorithms.md",
                                 "Util" => "lib/Util.md"],
                "Bibliography" => "bibliography.md",
                "About" => "about.md"])

deploydocs(; repo="github.com/JuliaReach/NeuralNetworkReachability.jl.git",
           push_preview=true)
