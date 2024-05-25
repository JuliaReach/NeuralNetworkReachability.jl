using Documenter, NeuralNetworkReachability

DocMeta.setdocmeta!(NeuralNetworkReachability, :DocTestSetup,
                    :(using NeuralNetworkReachability); recursive=true)

makedocs(; sitename="NeuralNetworkReachability.jl",
         modules=[NeuralNetworkReachability],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
                                collapselevel=1, assets=["assets/aligned.css"]),
         pagesonly=true,
         pages=["Home" => "index.md",
                "Library" => Any["ForwardAlgorithms" => "lib/ForwardAlgorithms.md",
                                 "BackwardAlgorithms" => "lib/BackwardAlgorithms.md",
                                 "BidirectionalAlgorithms" => "lib/BidirectionalAlgorithms.md",
                                 "Util" => "lib/Util.md"],
                "About" => "about.md"])

deploydocs(; repo="github.com/JuliaReach/NeuralNetworkReachability.jl.git",
           push_preview=true)
