export Dynamics,
    dynamics,
    generateDynamics,
    computePossibleSales,
    computeRewirings,
    computeTurn!,
    computeRound!,
    computeDynamic!, computeUnknownFirms

struct Dynamics{N}

    state::EconomyState{N}
    # eigval_M_Matrix::Array{Float64,1}
    # salesTimeSeries::Array{Array{Float64, 1}, 1}
    rewiringTimeSeries::Array{Int,1}

end

function dynamics(state::EconomyState{N}) where {N}
    d = length(state.economy.productivity)
    # eigvals_M_Matrix = Array{Float64,1}(undef, 0)
    # salesTimeSeries = Array{Array{Float64, 1}, 1}(undef, 0)
    rewiringTimeSeries = Array{Int,1}(undef, 0)
    return Dynamics(state, rewiringTimeSeries)
end

function generateDynamics(productivity, labourIntensity, technology, supply, foresight = -1)


    economy = LeontiefEconomy(productivity, labourIntensity, technology, foresight)

    sales, prices, quantities = computeEquilibrium(economy, supply)

    state = EconomyState(economy, supply, sales, prices, quantities)

    return dynamics(state)
end

function _computeKnownFirms(firm::Int64, foresight::Int64, supply::Matrix{Float64})
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the known firms
    knownFirms = zeros(Int64, 1)
    knownFirms[1] = firm
    if foresight == -1
        return collect(1:numberOfFirms)
    elseif foresight == 0
        return knownFirms
    else
        for j = 1:numberOfFirms
            if supply[firm, j] > 0 && j != firm
                push!(knownFirms, j)
            elseif supply[j, firm] > 0 && j != firm
                push!(knownFirms, j)
            end
        end
        for _ = 2:foresight
            for j in knownFirms
                for k = 1:numberOfFirms
                    if !in(k, knownFirms)
                        if supply[j, k] > 0 && k != j && !in(k, knownFirms)
                            push!(knownFirms, k)
                        end
                        if supply[k, j] > 0 && k != j && !in(k, knownFirms)
                            push!(knownFirms, k)
                        end
                    end
                end
            end
        end
        return knownFirms
    end
end

"""
define a function that given a number of firms and a vector of known firms
    it computes the unknown firms
"""
function _computeUnknownFirms(numberOfFirms::Int64, knownFirms::Vector{Int64})
    unknownFirms = zeros(Int64, 0)
    for i = 1:numberOfFirms
        if !in(i, knownFirms)
            push!(unknownFirms, i)
        end
    end
    return unknownFirms
end


function computeUnknownFirms(supplyChain, firm, foresight)
    knownFirms = _computeKnownFirms(firm, foresight, supplyChain)
    return _computeUnknownFirms(size(supplyChain, 1), knownFirms)
end


function computePossibleSales(firm, rewiring, state, foresight)

    oldWeight = state.supplyChain[floor(Int64, rewiring[1]), firm]
    state.supplyChain[rewiring[1], firm] = 0
    state.supplyChain[rewiring[2], firm] = rewiring[3]

    if foresight == -1
        retVal = computeEquilibrium(state.economy, state.supplyChain)[1][firm]
    else
        unknown = computeUnknownFirms(state.supplyChain, firm, foresight)
        retVal = computeEquilibrium(state, state.supplyChain, unknown, state.prices, state.quantities)[firm]
    end

    state.supplyChain[rewiring[1], firm] = oldWeight
    state.supplyChain[rewiring[2], firm] = 0

    return retVal
end

function computeRewirings(state, firm)
    supply = @views state.supplyChain
    technology = @views state.economy.technology
    rewirings = Array{Tuple{Int64,Int64,Float64},1}(undef, 0)
    push!(rewirings, (firm, firm, 0.0))
    n = length(state.supplyChain[:, firm])
    for i = 1:n
        if supply[i, firm] != 0
            for k = 1:n
                if technology[k, firm] != 0 && supply[k, firm] == 0
                    push!(rewirings, (i, k, technology[k, firm]))
                end
            end
        end
    end
    return rewirings
end

function computeTurn!(dynamics::Dynamics, firm::Int, discount = 1.)

    hasRewired = false
    possibleRewirings = computeRewirings(dynamics.state, firm)
    currentSales = dynamics.state.sales[firm]
    currentSales *= (1/discount)
    currentRewiring = possibleRewirings[1]
    for possibleRewiring in possibleRewirings[2:end]
        possibleSales = computePossibleSales(firm, possibleRewiring, dynamics.state, dynamics.state.economy.foresight)
        if possibleSales > currentSales
            currentSales = possibleSales
            currentRewiring = possibleRewiring
            hasRewired = true
        end
    end

    dynamics.state.supplyChain[currentRewiring[1], firm] = 0
    dynamics.state.supplyChain[currentRewiring[2], firm] = currentRewiring[3]

    computeEquilibrium!(dynamics.state)

    push!(dynamics.rewiringTimeSeries, (hasRewired ? firm : 0))

    return hasRewired

end


function computeRound!(dynamics::Dynamics, discount = 1.)

    hasChanged = false
    firms = collect(1:length(dynamics.state.sales))
    shuffle!(firms)
    for firm in firms
        hasChanged = (computeTurn!(dynamics, firm, discount) ? true : hasChanged)
    end
    # push!(dynamics.salesTimeSeries, deepcopy(dynamics.state.sales))
    # #         decomp, _ =
    # partialschur(sparse(-J + diagm(noise)), nev = 1, tol = 1e-6, which = SR())
    # results[i] = -minimum(real(decomp.eigenvalues))
    # push!(dynamics.eigval_M_Matrix, get_M_Matrix_eig(dynamics.state))

    return hasChanged

end

function computeDynamic!(dynamics::Dynamics, numberOfRounds::Int64)

    hasNotConverged = true
    for i = 1:numberOfRounds
        hasNotConverged = computeRound!(dynamics)
        if !hasNotConverged
            break
        end
    end
    return hasNotConverged

end
