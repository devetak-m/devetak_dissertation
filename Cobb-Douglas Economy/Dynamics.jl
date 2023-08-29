include("Firms.jl")
include("DynamicsType.jl")
using Random
"""
define a dynamics object
"""

struct DynamicsTimeSeries

    numberOfRounds::Int64
    economyState::EconomyState
    currentRound::Vector{Int64}
    # time series
    profitTimeSeries::Array{Float64,2}
    priceTimeSeries::Array{Float64,2}
    salesTimeSeries::Array{Float64,2}
    wageTimeSeries::Array{Float64,1}
    employmentTimeSeries::Array{Float64,2}
    supplyTimeSeries::Array{Float64,3}
    rewiringTimeSeries::Array{Float64,1}

    function DynamicsTimeSeries(economyState, numberOfRounds = 100)
        numberOfFirms = economyState.economy.numberOfFirms
        new(
            numberOfRounds,
            economyState,
            [0],
            Matrix{Float64}(undef, numberOfRounds * numberOfFirms, numberOfFirms),
            Matrix{Float64}(undef, numberOfRounds * numberOfFirms, numberOfFirms),
            Matrix{Float64}(undef, numberOfRounds * numberOfFirms, numberOfFirms),
            zeros(numberOfRounds * numberOfFirms),
            Matrix{Float64}(undef, numberOfRounds * numberOfFirms, numberOfFirms),
            Array{Float64}(undef, numberOfRounds, numberOfFirms, numberOfFirms),
            zeros(numberOfRounds * numberOfFirms),
        )
    end
end

function computeRewirings(economyState::EconomyState, firm::Int64)
    rewirings = nothing
    numberOfFirms = economyState.economy.numberOfFirms
    supply = economyState.supply
    technology = economyState.economy.technology
    for i = 1:numberOfFirms
        if supply[i, firm] != 0
            for k = 1:numberOfFirms
                if technology[k, firm] != 0 && supply[k, firm] == 0
                    if rewirings === nothing
                        rewirings = [i; k; technology[k, firm]]
                        rewirings = reshape(rewirings, (3, 1))
                    else
                        rewirings = [rewirings [i; k; technology[k, firm]]]
                    end
                end
            end
        end
    end
    if rewirings === nothing
        println("no rewiring possible for firm $(firm)!")
        println("supply: $(supply[:, firm])")
        println("technology: $(technology[:, firm])")
    end
    return rewirings
end

function computeRound!(dynamicsTimeSeries::DynamicsTimeSeries, tolerance = 1)

    # update the current round
    dynamicsTimeSeries.currentRound[1] += 1
    currentTurn = 1
    flagChange = false
    firmsToUpdate = collect(1:dynamicsTimeSeries.economyState.economy.numberOfFirms)
    shuffle!(firmsToUpdate)

    for firm in firmsToUpdate
        possibleRewirings = computeRewirings(dynamicsTimeSeries.economyState, firm)
        currentProfit = dynamicsTimeSeries.economyState.profits[firm]
        currentProfit *= (1/ tolerance)
        currentRewiring = [firm firm 0]
        currentSupply = copy(dynamicsTimeSeries.economyState.supply)
        flagChangeLocal = false
        if possibleRewirings != nothing
            for rewiring in eachcol(possibleRewirings)
                oldSupplier = floor(Int64, rewiring[1])
                newSupplier = floor(Int64, rewiring[2])
                capacityNewSupplier = rewiring[3]
                capacityOldSupplier = dynamicsTimeSeries.economyState.supply[oldSupplier, firm]
                currentSupply[oldSupplier, firm] = 0
                currentSupply[newSupplier, firm] = capacityNewSupplier
                newProfit =
                    computeExpectedProfits(dynamicsTimeSeries.economyState, currentSupply, firm)
                if newProfit > currentProfit
                    currentProfit = newProfit
                    currentRewiring = rewiring
                    flagChange = true
                    flagChangeLocal = true
                    dynamicsTimeSeries.rewiringTimeSeries[(dynamicsTimeSeries.currentRound[1]-1)*dynamicsTimeSeries.economyState.economy.numberOfFirms+currentTurn] =
                        copy(firm)
                end

                currentSupply[oldSupplier, firm] = capacityOldSupplier
                currentSupply[newSupplier, firm] = 0
            end
        end

        if flagChangeLocal
            dynamicsTimeSeries.economyState.supply[floor(Int64, currentRewiring[1]), firm] =
                0
            dynamicsTimeSeries.economyState.supply[floor(Int64, currentRewiring[2]), firm] =
                currentRewiring[3]
            computeEquilibrium!(dynamicsTimeSeries.economyState)
        end


        # update the time series
        currentPosition =
            (dynamicsTimeSeries.currentRound[1] - 1) *
            dynamicsTimeSeries.economyState.economy.numberOfFirms + currentTurn
        dynamicsTimeSeries.profitTimeSeries[currentPosition, :] =
            copy(dynamicsTimeSeries.economyState.profits)
        dynamicsTimeSeries.priceTimeSeries[currentPosition, :] =
            copy(dynamicsTimeSeries.economyState.prices)
        dynamicsTimeSeries.salesTimeSeries[currentPosition, :] =
            copy(dynamicsTimeSeries.economyState.sales)
        dynamicsTimeSeries.wageTimeSeries[currentPosition, :] =
            copy(dynamicsTimeSeries.economyState.wage)
        dynamicsTimeSeries.employmentTimeSeries[currentPosition, :] =
            copy(dynamicsTimeSeries.economyState.employment)

        currentTurn += 1
    end
    dynamicsTimeSeries.supplyTimeSeries[dynamicsTimeSeries.currentRound[1], :, :] =
        copy(dynamicsTimeSeries.economyState.supply)

    return flagChange
end

function computeDynamics!(
    dynamicsTimeSeries::DynamicsTimeSeries,
    dynamicsType::DeterministicDynamics,
)
    flagChange = true
    dynamicsTimeSeries.supplyTimeSeries[1, :, :] =
        copy(dynamicsTimeSeries.economyState.supply)
    while flagChange &&
        dynamicsTimeSeries.currentRound[1] < dynamicsTimeSeries.numberOfRounds
        flagChange = computeRound!(dynamicsTimeSeries, dynamicsType.tol)
    end
    return dynamicsTimeSeries.currentRound[1] != dynamicsTimeSeries.numberOfRounds
end

function computeDynamics!(
    dynamicsTimeSeries::DynamicsTimeSeries,
    dynamicsType::RandomDynamics,
)
    probability = dynamicsType.prob
    for j = 1:dynamicsTimeSeries.numberOfRounds
        for firm = 1:dynamicsTimeSeries.economyState.economy.numberOfFirms
            if rand() < probability
                rewiring = computeRewirings(dynamicsTimeSeries.economyState, firm)
                rewiring = rewiring[:, rand(1:size(rewiring, 2))]
                dynamicsTimeSeries.economyState.supply[floor(Int64, rewiring[1]), firm] = 0
                dynamicsTimeSeries.economyState.supply[floor(Int64, rewiring[2]), firm] =
                    rewiring[3]
            end
        end
        # store the time series of the supply matrix
        dynamicsTimeSeries.supplyTimeSeries[j, :, :] =
            copy(dynamicsTimeSeries.economyState.supply)
        dynamicsTimeSeries.currentRound[1] += 1
    end
end

function computeHouseholdUtility(prices)
    return -sum(log.(prices))
end
