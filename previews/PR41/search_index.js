var documenterSearchIndex = {"docs":
[{"location":"lib/ForwardAlgorithms/#ForwardAlgorithms","page":"ForwardAlgorithms","title":"ForwardAlgorithms","text":"","category":"section"},{"location":"lib/ForwardAlgorithms/","page":"ForwardAlgorithms","title":"ForwardAlgorithms","text":"This section of the manual describes the module for forward algorithms.","category":"page"},{"location":"lib/ForwardAlgorithms/","page":"ForwardAlgorithms","title":"ForwardAlgorithms","text":"Pages = [\"ForwardAlgorithms.md\"]\nDepth = 3","category":"page"},{"location":"lib/ForwardAlgorithms/","page":"ForwardAlgorithms","title":"ForwardAlgorithms","text":"CurrentModule = NeuralNetworkReachability.ForwardAlgorithms","category":"page"},{"location":"lib/ForwardAlgorithms/","page":"ForwardAlgorithms","title":"ForwardAlgorithms","text":"ForwardAlgorithm\nDefaultForward\nConcreteForward\nLazyForward\nBoxForward\nSplitForward\nDeepZ\nAI2Box\nAI2Zonotope\nAI2Polytope\nPolyZonoForward\nVerisig","category":"page"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.ForwardAlgorithm","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.ForwardAlgorithm","text":"ForwardAlgorithm\n\nAbstract supertype of forward algorithms.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.DefaultForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.DefaultForward","text":"DefaultForward <: ForwardAlgorithm\n\nDefault forward algorithm, which works for vector-like inputs.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.ConcreteForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.ConcreteForward","text":"ConcreteForward <: ForwardAlgorithm\n\nForward algorithm that uses concrete set operations.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.LazyForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.LazyForward","text":"LazyForward <: ForwardAlgorithm\n\nForward algorithm that uses lazy set operations.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.BoxForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.BoxForward","text":"BoxForward{AMA<:ForwardAlgorithm} <: ForwardAlgorithm\n\nForward algorithm that uses a box approximation for non-identity activations and applies the affine map according to the specified algorithm.\n\nFields\n\naffine_map_algorithm – algorithm to apply for affine maps\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.SplitForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.SplitForward","text":"SplitForward{S<:ForwardAlgorithm,FS,FM} <: ForwardAlgorithm\n\nForward algorithm that splits a set, then computes the image under the neural network, and finally merges the resulting sets again, all according to a policy.\n\nFields\n\nalgo – algorithm to be applied between splitting and merging\nsplit_function – function for splitting\nmerge_function – function for merging\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.DeepZ","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.DeepZ","text":"DeepZ <: ForwardAlgorithm\n\nForward algorithm based on zonotopes for ReLU, sigmoid, and tanh activation functions from [1].\n\n[1]: Singh et al.: Fast and Effective Robustness Certification, NeurIPS 2018.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.AI2Box","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.AI2Box","text":"AI2Box <: AI2\n\nAI2 forward algorithm for ReLU activation functions based on abstract interpretation with the interval domain from [1].\n\nNotes\n\nThis algorithm is less precise than BoxForward because it abstracts after every step, including the affine map.\n\n[1]: Gehr et al.: AI²: Safety and robustness certification of neural networks with abstract interpretation, SP 2018.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.AI2Zonotope","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.AI2Zonotope","text":"AI2Zonotope <: AI2\n\nAI2 forward algorithm for ReLU activation functions based on abstract interpretation with the zonotope domain from [1].\n\nFields\n\njoin_algorithm – (optional; default: \"join\") algorithm to compute the                     join of two zonotopes\n\n[1]: Gehr et al.: AI²: Safety and robustness certification of neural networks with abstract interpretation, SP 2018.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.AI2Polytope","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.AI2Polytope","text":"AI2Polytope <: AI2\n\nAI2 forward algorithm for ReLU activation functions based on abstract interpretation with the polytope domain from [1].\n\n[1]: Gehr et al.: AI²: Safety and robustness certification of neural networks with abstract interpretation, SP 2018.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.PolyZonoForward","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.PolyZonoForward","text":"PolyZonoForward{A<:PolynomialApproximation,N,R} <: ForwardAlgorithm\n\nForward algorithm based on poynomial zonotopes via polynomial approximation from [1].\n\nFields\n\npolynomial_approximation – method for polynomial approximation\nreduced_order          – order to which the result will be reduced after                               each layer\ncompact                  – predicate for compacting the result after each                               layer\n\nNotes\n\nThe default constructor takes keyword arguments with the following defaults:\n\npolynomial_approximation: RegressionQuadratic(10), i.e., quadratic regression with 10 samples\ncompact: () -> true, i.e., compact after each layer\n\nSee the subtypes of PolynomialApproximation for available polynomial approximation methods.\n\n[1]: Kochdumper et al.: Open- and closed-loop neural network verification using polynomial zonotopes, NFM 2023.\n\n\n\n\n\n","category":"type"},{"location":"lib/ForwardAlgorithms/#NeuralNetworkReachability.ForwardAlgorithms.Verisig","page":"ForwardAlgorithms","title":"NeuralNetworkReachability.ForwardAlgorithms.Verisig","text":"Verisig{R} <: ForwardAlgorithm\n\nForward algorithm for sigmoid and tanh activation functions from [1].\n\nFields\n\nalgo – reachability algorithm of type TMJets\n\nNotes\n\nThe implementation is known to be unsound in some cases.\n\nThe implementation currently only supports neural networks with a single hidden layer.\n\n[1] Ivanov et al.: Verisig: verifying safety properties of hybrid systems with neural network controllers, HSCC 2019.\n\n\n\n\n\n","category":"type"},{"location":"about/#About","page":"About","title":"About","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"This page contains some general information about this project and recommendations about contributing.","category":"page"},{"location":"about/","page":"About","title":"About","text":"Pages = [\"about.md\"]","category":"page"},{"location":"about/#Contributing","page":"About","title":"Contributing","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"If you like this package, consider contributing!","category":"page"},{"location":"about/","page":"About","title":"About","text":"Creating an issue in the NeuralNetworkReachability GitHub issue tracker to report a bug, open a discussion about existing functionality, or suggest new functionality is appreciated.","category":"page"},{"location":"about/","page":"About","title":"About","text":"If you have written code and would like it to be peer reviewed and added to the library, you can fork the repository and send a pull request (see below).","category":"page"},{"location":"about/","page":"About","title":"About","text":"You are also welcome to get in touch with us in the JuliaReach Zulip channel.","category":"page"},{"location":"about/","page":"About","title":"About","text":"Below we give some general comments about contributing to this package. The JuliaReach development documentation describes coding guidelines; take a look when in doubt about the coding style that is expected for the code that is finally merged into the library.","category":"page"},{"location":"about/#Branches-and-pull-requests-(PR)","page":"About","title":"Branches and pull requests (PR)","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"We use a standard pull-request policy: You work in a private branch and eventually add a pull request, which is then reviewed by other programmers and merged into the master branch.","category":"page"},{"location":"about/","page":"About","title":"About","text":"Each pull request should be based on a branch with the name of the author followed by a descriptive name, e.g., mforets/my_feature. If the branch is associated to a previous discussion in an issue, we use the number of the issue for easier lookup, e.g., mforets/7.","category":"page"},{"location":"about/#Unit-testing-and-continuous-integration-(CI)","page":"About","title":"Unit testing and continuous integration (CI)","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"This project is synchronized with GitHub Actions such that each PR gets tested before merging (and the build is automatically triggered after each new commit). For the maintainability of this project, it is important to make all unit tests pass.","category":"page"},{"location":"about/","page":"About","title":"About","text":"To run the unit tests locally, you can do:","category":"page"},{"location":"about/","page":"About","title":"About","text":"julia> using Pkg\n\njulia> Pkg.test(\"NeuralNetworkReachability\")","category":"page"},{"location":"about/","page":"About","title":"About","text":"We also advise adding new unit tests when adding new features to ensure long-term support of your contributions.","category":"page"},{"location":"about/#Contributing-to-the-documentation","page":"About","title":"Contributing to the documentation","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"New functions and types should be documented according to the JuliaReach development documentation.","category":"page"},{"location":"about/","page":"About","title":"About","text":"You can view the source-code documentation from inside the REPL by typing ? followed by the name of the type or function.","category":"page"},{"location":"about/","page":"About","title":"About","text":"The documentation you are currently reading is written in Markdown, and it relies on the package Documenter.jl to produce the final layout. The sources for creating this documentation are found in docs/src. You can easily include the documentation that you wrote for your functions or types there (see the source code or Documenter's guide for examples).","category":"page"},{"location":"about/","page":"About","title":"About","text":"To generate the documentation locally, run docs/make.jl, e.g., by executing the following command in the terminal:","category":"page"},{"location":"about/","page":"About","title":"About","text":"$ julia --color=yes docs/make.jl","category":"page"},{"location":"about/#Credits","page":"About","title":"Credits","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"Here we list the names of the maintainers of the NeuralNetworkReachability.jl library, as well as past and present contributors (in alphabetic order).","category":"page"},{"location":"about/#Core-developers","page":"About","title":"Core developers","text":"","category":"section"},{"location":"about/","page":"About","title":"About","text":"Marcelo Forets, Universidad de la República\nChristian Schilling, Aalborg University","category":"page"},{"location":"lib/BackwardAlgorithms/#BackwardAlgorithms","page":"BackwardAlgorithms","title":"BackwardAlgorithms","text":"","category":"section"},{"location":"lib/BackwardAlgorithms/","page":"BackwardAlgorithms","title":"BackwardAlgorithms","text":"This section of the manual describes the module for backward algorithms.","category":"page"},{"location":"lib/BackwardAlgorithms/","page":"BackwardAlgorithms","title":"BackwardAlgorithms","text":"Pages = [\"BackwardAlgorithms.md\"]\nDepth = 3","category":"page"},{"location":"lib/BackwardAlgorithms/","page":"BackwardAlgorithms","title":"BackwardAlgorithms","text":"CurrentModule = NeuralNetworkReachability.BackwardAlgorithms","category":"page"},{"location":"lib/BackwardAlgorithms/","page":"BackwardAlgorithms","title":"BackwardAlgorithms","text":"BackwardAlgorithm\nPolyhedraBackward\nBoxBackward\nPartitioningLeakyReLU","category":"page"},{"location":"lib/BackwardAlgorithms/#NeuralNetworkReachability.BackwardAlgorithms.BackwardAlgorithm","page":"BackwardAlgorithms","title":"NeuralNetworkReachability.BackwardAlgorithms.BackwardAlgorithm","text":"BackwardAlgorithm\n\nAbstract supertype of backward algorithms.\n\n\n\n\n\n","category":"type"},{"location":"lib/BackwardAlgorithms/#NeuralNetworkReachability.BackwardAlgorithms.PolyhedraBackward","page":"BackwardAlgorithms","title":"NeuralNetworkReachability.BackwardAlgorithms.PolyhedraBackward","text":"PolyhedraBackward <: BackwardAlgorithm\n\nBackward algorithm for piecewise-affine activations; uses a union of polyhedra.\n\n\n\n\n\n","category":"type"},{"location":"lib/BackwardAlgorithms/#NeuralNetworkReachability.BackwardAlgorithms.BoxBackward","page":"BackwardAlgorithms","title":"NeuralNetworkReachability.BackwardAlgorithms.BoxBackward","text":"BoxBackward <: BackwardAlgorithm\n\nBackward algorithm that uses a polyhedral approximation with axis-aligned linear constraints.\n\n\n\n\n\n","category":"type"},{"location":"lib/BackwardAlgorithms/#NeuralNetworkReachability.BackwardAlgorithms.PartitioningLeakyReLU","page":"BackwardAlgorithms","title":"NeuralNetworkReachability.BackwardAlgorithms.PartitioningLeakyReLU","text":"PartitioningLeakyReLU{N<:Real}\n\nIterator over the partitions of a leaky ReLU activation.\n\nFields\n\nn     – dimension\nslope – slope of the leaky ReLU activation\n\n\n\n\n\n","category":"type"},{"location":"lib/BidirectionalAlgorithms/#BidirectionalAlgorithms","page":"BidirectionalAlgorithms","title":"BidirectionalAlgorithms","text":"","category":"section"},{"location":"lib/BidirectionalAlgorithms/","page":"BidirectionalAlgorithms","title":"BidirectionalAlgorithms","text":"This section of the manual describes the module for bidirectional algorithms.","category":"page"},{"location":"lib/BidirectionalAlgorithms/","page":"BidirectionalAlgorithms","title":"BidirectionalAlgorithms","text":"Pages = [\"BidirectionalAlgorithms.md\"]\nDepth = 3","category":"page"},{"location":"lib/BidirectionalAlgorithms/","page":"BidirectionalAlgorithms","title":"BidirectionalAlgorithms","text":"CurrentModule = NeuralNetworkReachability.BidirectionalAlgorithms","category":"page"},{"location":"lib/BidirectionalAlgorithms/","page":"BidirectionalAlgorithms","title":"BidirectionalAlgorithms","text":"BidirectionalAlgorithm\nSimpleBidirectional","category":"page"},{"location":"lib/BidirectionalAlgorithms/#NeuralNetworkReachability.BidirectionalAlgorithms.BidirectionalAlgorithm","page":"BidirectionalAlgorithms","title":"NeuralNetworkReachability.BidirectionalAlgorithms.BidirectionalAlgorithm","text":"BidirectionalAlgorithm\n\nAbstract supertype of bidirectional algorithms.\n\n\n\n\n\n","category":"type"},{"location":"lib/BidirectionalAlgorithms/#NeuralNetworkReachability.BidirectionalAlgorithms.SimpleBidirectional","page":"BidirectionalAlgorithms","title":"NeuralNetworkReachability.BidirectionalAlgorithms.SimpleBidirectional","text":"SimpleBidirectional{FA<:ForwardAlgorithm, BA<:BackwardAlgorithm} <: BidirectionalAlgorithm\n\nSimple bidirectional algorithm parametric in a forward and backward algorithm.\n\nFields\n\nfwd_algo – forward algorithm\nbwd_algo – backward algorithm\n\n\n\n\n\n","category":"type"},{"location":"lib/Util/#Util","page":"Util","title":"Util","text":"","category":"section"},{"location":"lib/Util/","page":"Util","title":"Util","text":"This section of the manual describes the module for utilities.","category":"page"},{"location":"lib/Util/","page":"Util","title":"Util","text":"Pages = [\"Util.md\"]\nDepth = 3","category":"page"},{"location":"lib/Util/","page":"Util","title":"Util","text":"CurrentModule = NeuralNetworkReachability.Util","category":"page"},{"location":"lib/Util/","page":"Util","title":"Util","text":"ConvSet","category":"page"},{"location":"lib/Util/#NeuralNetworkReachability.Util.ConvSet","page":"Util","title":"NeuralNetworkReachability.Util.ConvSet","text":"ConvSet{T<:LazySet{N}}\n\nWrapper of a set to represent a three-dimensional structure.\n\nFields\n\nset  – set of dimension dims[1] * dims[2] * dims[3]\ndims – 3-tuple with the dimensions\n\n\n\n\n\n","category":"type"},{"location":"#NeuralNetworkReachability.jl","page":"Home","title":"NeuralNetworkReachability.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"NeuralNetworkReachability.jl is a Julia package for reachability analysis of artificial neural networks.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\"]","category":"page"}]
}
