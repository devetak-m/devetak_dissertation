export get_M_Matrix, getHarfindalIndex, getSupplyMatrix, getRegularMatrix, getEmpiricalMatrix, findCommunities, adjustedMutualInformation, generateVector, computeOverlap, getHomEconomy, getHetEconomy

function get_M_Matrix(state::EconomyState)
    am = 1 .- state.economy.labourIntensity
    return diagm(Vector(state.economy.productivity)) - (am' .* state.supplyChain)'
end

function getHarfindalIndex(vector)
    return sum(vector.^4)
end

function getSupplyMatrix(technologyMatrix)

    supplyMatrix = deepcopy(technologyMatrix)
    # in each column calculate the sum of the elements
    columnSum = sum(technologyMatrix, dims = 1)
    # for every column that has more than two nonzero elements
    # select half of them randmoly and set them to zero
    for i in axes(technologyMatrix, 2)
        if columnSum[i] > 2
            indices = findall(x -> x != 0, technologyMatrix[:, i])
            shuffle!(indices)
            indices = indices[1:Int(floor(columnSum[i] / 2))]
            for j in indices
                supplyMatrix[j, i] = 0
            end
        elseif columnSum[i] == 0
            # add two entries in the technology matrix
            # one of which is a supplier
            indices = findall(x -> x == 0, technologyMatrix[:, i])
            shuffle!(indices)
            indices = indices[1:2]
            technologyMatrix[indices[1], i] = 1
            technologyMatrix[indices[2], i] = 1
            supplyMatrix[indices[2], i] = 1
        else
            # add an extra supplier
            indices = findall(x -> x == 0, technologyMatrix[:, i])
            shuffle!(indices)
            index = indices[1]
            technologyMatrix[index, i] = 1
        end
    end
    # normalize every column so that it sums to one
    for i in axes(technologyMatrix, 2)
        factor = sum(supplyMatrix[:, i])
        supplyMatrix[:, i] = supplyMatrix[:, i] ./ factor
        technologyMatrix[:, i] = technologyMatrix[:, i] ./ factor
    end

    return supplyMatrix
end

function getRegularMatrix(size::Int64, connectivity)
    graph = random_regular_digraph(size, connectivity)
    return transpose(Matrix((adjacency_matrix(graph)))) ./ 1.0
end

function getEmpiricalMatrix(size, connectivity)
    return transpose(
        Matrix(adjacency_matrix(static_scale_free(size, connectivity * size, 2.0, 2.0))),
    ) ./ 1.0
end

"""
    findCommunities(matrix)

TBW
"""
function findCommunities(matrix, maxIter = 100000, communities = 3)

    # clean the matrix and construct the laplacian
    M = matrix + matrix'
    M[M .!= 0] .= 1
    L = diagm(sum(M, dims=1)[:]) - M

    # get the eigenvectors
    decomp, _ = partialschur(L, nev=communities, tol=1e-6, which=SR())
    eigenvectors = decomp.Q[:, 1:communities]
    # eigenvectors[eigenvectors .< 0] .= -1
    # eigenvectors[eigenvectors .> 0] .= 1

    # cluster and find the most equilibrated solution
    oldfit = 10000.
    cluesters = Array{Array{Int64,1},1}(undef, 3)
    R = kmeans(eigenvectors', communities, maxiter=1)
    a = R.assignments
    for i in 1:3
        cluesters[i] = findall(a .== i)
    end
    for i in 1:maxIter
        Random.seed!(i)
        R = kmeans(eigenvectors', communities, maxiter=8000, init=:rand)
        fit = sum(abs.(R.counts .- size(matrix, 1) / communities))
        if fit < oldfit
            oldfit = fit
            for i in 1:3
                cluesters[i] = findall(a .== i)
            end
        end
    end
    # sort by lenght
    cluesters = sort(cluesters, by=length, rev=true)
    return cluesters
end



function expectedMutualInformation(U, V)
    R = length(U)
    C = length(V)
    N = BigInt(length([(U...)...]))
    nMatrix = zeros(Int64 ,R, C)
    for i in axes(nMatrix, 1)
        for j in axes(nMatrix, 2)
            nMatrix[i, j] = length(intersect(U[i], V[j]))
        end
    end

    aVector = BigInt.(sum(nMatrix, dims = 2)[:, 1])
    bVector = BigInt.(sum(nMatrix, dims = 1)[1, :])
    expectedMutualInformation = 0.0

    for i in 1:R
        for j in 1:C
            bottom = max(aVector[i] + bVector[j] - N, 0)
            for nij in bottom:min(aVector[i], bVector[j])
                if nij == 0
                    continue
                end
                prod = factorial(BigInt(N - aVector[i])) / (factorial(BigInt(aVector[i] - nij)) * factorial(BigInt(bVector[j] - nij)))
                expectedMutualInformation += 
                    nij / N * log(N * nij / (aVector[i] * bVector[j])) *
                    (fallingfactorial(bVector[j], nij) * fallingfactorial(N - bVector[j], N - aVector[i] - bVector[j] + nij))/
                    (fallingfactorial(N, aVector[i])) * prod
            end
        end
    end
    @show expectedMutualInformation
    return expectedMutualInformation
                
end

function mutualInformation(U, V)
    R = length(U)
    C = length(V)
    N = length([(U...)...])
    nMatrix = zeros(Int64 ,R, C)
    for i in axes(nMatrix, 1)
        for j in axes(nMatrix, 2)
            nMatrix[i, j] = length(intersect(U[i], V[j]))
        end
    end
    sizeU = length.(U)
    sizeV = length.(V)


    mutualInformation = 0.0

    for i in 1:R
        for j in 1:C
            if nMatrix[i, j] == 0
                continue
            end
            mutualInformation += 
                nMatrix[i, j] / N * log(N * nMatrix[i, j] / (sizeU[i] * sizeV[j]))
        end
    end

    return mutualInformation

end

function entropyOfClustering(U)
    R = length(U)
    N = length([(U...)...])
    sizeU = length.(U)
    entropyOfClustering = 0.0
    for i in 1:R
        entropyOfClustering += sizeU[i] / N * log(sizeU[i] / N)
    end
    return - entropyOfClustering
end


function adjustedMutualInformation(U, V)
    MI = mutualInformation(U, V)
    entropyU = entropyOfClustering(U)
    entropyV = entropyOfClustering(V)
    EMI = expectedMutualInformation(U, V)
    return (MI - EMI) / (max(entropyU, entropyV) - EMI)
end

function fallingfactorial(n, r)
    if n == r
        return one(typeof(n))
    else
        return n * fallingfactorial(n-1, r)
    end
end

function generateVector(technology, supply)
    idx_active = findall(x -> x != 0, technology)
    return_vector = zeros(length(idx_active))
    for i in eachindex(idx_active)
        idx = idx_active[i]
        if supply[idx] == 0
            return_vector[i] = -1
        else
            return_vector[i] = 1
        end
    end
    return return_vector
end

function computeOverlap(vector1, vector2)
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))
end

function getHomEconomy(numberOfFirms, technologyMatrix, supplyMatrix, numberOfRounds, t)

    a = ones(numberOfFirms) .* 0.3
    p = ones(numberOfFirms) * 1.5

    return generateDynamics(p, a, technologyMatrix, supplyMatrix, t)

end

function getHetEconomy(numberOfFirms, technologyMatrix, supplyMatrix, numberOfRounds, t)

    a = ones(numberOfFirms) .* 0.3
    p = rand(numberOfFirms) * 3.

    return generateDynamics(p, a, technologyMatrix, supplyMatrix, t)

end