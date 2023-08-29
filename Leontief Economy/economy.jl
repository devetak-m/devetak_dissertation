export LeontiefEconomy, EconomyState, computeEquilibrium!, computeEquilibrium, generateKnownMatrix

struct LeontiefEconomy{P}
    productivity::AbstractVector{P}
    labourIntensity::AbstractVector{P}
    technology::AbstractMatrix{P}
    foresight::Int
end

struct EconomyState{P}

    economy::LeontiefEconomy{P}
    supplyChain::Matrix{P}
    sales::AbstractVector{P}
    prices::AbstractVector{P}
    quantities::AbstractVector{P}

end

function computeEquilibrium(economy::LeontiefEconomy, supplyChain)

    am = 1 .- economy.labourIntensity

    M = diagm(economy.productivity) - (am' .* supplyChain)' # ?! M_ij = z_id_ij - (1-a_j)*w_ij

    p_t = M \ economy.labourIntensity

    # M_tilde = diagm(Vector(economy.productivity)) - supplyChain .* am' is M^T

    g_t = M' \ (1 ./p_t .* 1 / length(p_t))

    h = sum(economy.labourIntensity' * g_t)/length(economy.labourIntensity)

    p = p_t .* h
    g = g_t ./ h

    return economy.productivity .* p .* g, p, g

end

function computeEquilibrium!(economy::EconomyState)


    s, p, q = computeEquilibrium(economy.economy, economy.supplyChain)

    economy.sales .= s
    economy.prices .= p
    economy.quantities .= q
    return nothing

end

function generateKnownMatrix(k, n)
    retVal = zeros(length(k), n)
    for i in eachindex(k)
        retVal[i, k[i]] = 1
    end
    return retVal
end

function computeEquilibrium(economy::EconomyState, supplyChain, known, prices, quantities)

    am = 1 .- economy.economy.labourIntensity

    M = diagm(economy.economy.productivity) - (am' .* supplyChain)' # ?! M_ij = z_id_ij - (1-a_j)*w_ij

    nM = M
    nM[known, :] .= generateKnownMatrix(known, length(am))
    am[known] .= prices[known]

    p_t = nM \ economy.economy.labourIntensity

    nMt = M'
    nMt[known, :] .= generateKnownMatrix(known, length(am))
    pn = 1 ./ p_t .* 1 / length(p_t)
    pn[known] .= quantities[known]
    g_t = nMt \ pn
    
    return economy.economy.productivity .* p_t .* g_t

end