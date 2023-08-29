"""
    findCommunities(matrix)

TBW
"""
function findCommunities(matrix)
    M = matrix + matrix'
    M[M .!= 0] .= 1
    L = diagm(sum(M, dims=1)[:]) - M
    decomp, _ = partialschur(L, nev=3, tol=1e-6, which=SR())f
    eigenvectors = decomp.Q[:, 1:3]
    oldfit = 10000.
    cluesters = Array{Array{Int64,1},1}(undef, 3)
    R = kmeans(eigenvectors', 3, maxiter=5000)
    a = R.assignments
    for i in 1:3
        cluesters[i] = findall(a .== i)
    end
    for i in 1:100000
        Random.seed!(i)
        R = kmeans(eigenvectors', 3, maxiter=8000, init=:rand)
        fit = sum(abs.(R.counts .- 40))
        if fit < oldfit
            oldfit = fit
            for i in 1:3
                cluesters[i] = findall(a .== i)
            end
        end
    end
    # sort by lenght
    cluesters = sort(cluesters, by=length)
    return cluesters
end