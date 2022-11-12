# FastKmeans

[![Build Status](https://travis-ci.org/lstagner/FastKmeans.jl.svg?branch=master)](https://travis-ci.org/lstagner/FastKmeans.jl)

This package implements a fast k-means algorithm[1]

>Hamerly, Greg. "Making k-means even faster." Proceedings of the 2010 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2010.

# Example

```julia
julia> using Clustering, FastKmeans, Base.Iterators

julia> x = range(-1,1,length=10)

julia> p = Vector{Float64}[]; sizehint!(p,20_000_000)

julia> for xc in product(x,x)
           xx = collect(xc)
           append!(p,[xx .+ 0.04*randn(2) for i=1:12500])
       end

julia> p_matrix = hcat(p...);

julia> @time a, centers = fastkmeans(p,100);
 1.982794 seconds (150 allocations: 104.917 MiB, 1.26% gc time)

julia> @time k = Clustering.kmeans(p_matrix,100);
 102.383593 seconds (743 allocations: 1.760 GiB, 0.42% gc time)
```
