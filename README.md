# FastKmeans

[![Build Status](https://travis-ci.org/lstagner/FastKmeans.jl.svg?branch=master)](https://travis-ci.org/lstagner/FastKmeans.jl)

This package implements a fast k-means algorithm[1]

>Hamerly, Greg. "Making k-means even faster." Proceedings of the 2010 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2010.

# Example

```julia
julia> using Clustering, FastKmeans, Iterators

julia> x = linspace(-1,1,10)

julia> p = Vector{Float64}[]; sizehint!(p,20_000_000)

julia> for xc in product(x,x)
           xx = collect(xc)
           append!(p,[xx + 0.04*randn(2) for i=1:12500])
       end

julia> p_matrix = hcat(p...);

julia> @time a, centers = fastkmeans(p,100);
 7.415347 seconds (126.25 M allocations: 5.718 GB, 11.89% gc time)

julia> @time k = Clustering.kmeans(p_matrix,100);
 54.080827 seconds (125.01 M allocations: 33.686 GB, 2.93% gc time)
```
