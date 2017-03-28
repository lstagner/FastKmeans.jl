module FastKmeans
# Hamerly, Greg. "Making k-means even faster."
# Proceedings of the 2010 SIAM international
# conference on data mining. Society for Industrial
# and Applied Mathematics, 2010.

using Clustering
using StaticArrays

function point_all_ctrs!(i,x,c,a,u,l)
    k = length(c)
    xx = x[i]
    cc = c[1]
    yy = xx - cc
    d1min = dot(yy,yy)
    d2min = typemax(typeof(d1min))
    ind = 1
    ai = a[i]
    @inbounds for j=1:k
        yy = xx - c[j]
        d1 = dot(yy,yy)
        d1 < d1min && (d1min = d1; ind = j)
        j == ai && continue
        d1 < d2min && (d2min = d1)
    end
    l[i] = sqrt(d2min)

    a[i] = ind
    u[i] = sqrt(d1min)
end

function initialize!(c,x,q,cp,u,l,a)
    @inbounds for i=1:length(x)
        point_all_ctrs!(i,x,c,a,u,l)
        ai = a[i]
        q[ai] += 1
        cp[ai] += x[i]
    end
end

function move_centers!(cp,q,c,p)
    k = length(c)
    pmax = zero(eltype(p))
    @inbounds for j=1:k
        cs = c[j]
        c[j] = cp[j]/q[j]
        pp = norm(cs-c[j])
        pp > pmax && (pmax=pp)
        p[j] = pp
    end
    return pmax
end

function update_bounds!(p,a,u,l)
    r = 1
    pmax = p[r]
    pmax2 = 0.0
    for j=1:length(p)
        p[j] > pmax && (pmax2 = pmax; pmax=p[j]; r = j)
    end

    @inbounds for i=1:length(u)
        u[i] += p[a[i]]
        r == a[i] ? (l[i] -= pmax2) : (l[i] -= pmax)
    end
end

function fastkmeans!(x,c, maxiter = 100)
    n = length(x)
    k = length(c)
    cp = c .* 0
    q = zeros(Int64,k)
    p = zeros(Float64,k)
    s = zeros(Float64,k)

    a = zeros(Int,n)
    l = zeros(Float64,n)
    u = zeros(Float64,n)

    initialize!(c,x,q,cp,u,l,a)

    for iter=1:maxiter
        @inbounds for j=1:k
            cc = c[j]
            dmin2 = typemax(eltype(cc))
            for jj=1:k
                jj == j && continue
                v = c[jj]-cc
                d = dot(v,v)
                d < dmin2 && (dmin2 = d)
            end
            s[j] = sqrt(dmin2)
        end

        @inbounds for i=1:n
            ap = a[i]
            xx = x[i]
            m = max(s[ap]/2,l[i])
            u[i] <= m && continue
            u[i] = norm(xx-c[ap])
            u[i] <= m && continue
            point_all_ctrs!(i,x,c,a,u,l)
            if ap != a[i]
                q[ap] -= 1
                q[a[i]] += 1
                cp[ap] -= xx
                cp[a[i]] += xx
            end
        end

        pmax = move_centers!(cp,q,c,p)
        pmax == 0.0 && break
        update_bounds!(p,a,u,l)
    end

    return a
end

function fastkmeans(x,k; maxiter = 100)
    x_matrix = hcat(x...)
    ci = Clustering.initseeds(:kmpp, x_matrix, k)
    x_s = SVector{length(x[1]),eltype(x[1])}.(x)
    c_s = SVector{length(x[1]),eltype(x[1])}.(x[ci])
    assignment = fastkmeans!(x_s, c_s, maxiter)

    return assignment, c_s
end

export fastkmeans

end # module
