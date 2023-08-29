using LinearAlgebra
export Firm, FirmState, Economy, EconomyState

"""
define an economy type struct which parametrizes the economy
    - const number of firms
    - const labour intensity vector
    - const productivity vector
    - const return to scale vector
    - const foresight vector
    - const technology matrix
"""
struct Economy
    numberOfFirms::Int64
    labourIntensity::Vector{Float64}
    productivity::Vector{Float64}
    returnToScale::Vector{Float64}
    foresight::Vector{Int64}
    technology::Matrix{Float64}
end

"""
define an economy state mutable struct which stores the state of the economy
    - const Economy
    - supply matrix
    - profits
    - prices
    - sales
    - wage
    - employment
"""
struct EconomyState
    economy::Economy
    supply::Matrix{Float64}
    profits::Vector{Float64}
    prices::Vector{Float64}
    sales::Vector{Float64}
    wage::Vector{Float64}
    employment::Vector{Float64}
end


"""
define a function that given an EconomyState it compute the supplyTilde_ij matrix defined as:
    supplyTilde_ij = (1 - labourIntensity_j) * returnToScale_j * supply_ij
"""

function _computeSupplyTildeMainLoop!(
    supplyTilde,
    supply,
    labourIntensity,
    returnToScale,
    numberOfFirms,
)
    @simd for i = 1:numberOfFirms
        @simd for j = 1:numberOfFirms
            @inbounds supplyTilde[j, i] =
                (1 - labourIntensity[i]) * returnToScale[i] * supply[j, i]
        end
    end
end

function _computeSupplyTilde(supply, labourIntensity, returnToScale)
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the supplyTilde matrix
    supplyTilde = similar(supply)
    _computeSupplyTildeMainLoop!(
        supplyTilde,
        supply,
        labourIntensity,
        returnToScale,
        numberOfFirms,
    )
    # compute the supplyTilde matrix
    # return the supplyTilde matrix
    return supplyTilde
end

"""
define a function that given an EconomyState it compute the sales as:
    sales = (I - supplyTilde_ij)^(-1) * (1/numberOfFirms)
"""
function _computeSales(supplyTilde, numberOfFirms)
    identityMatrix = Matrix{Float64}(I, numberOfFirms, numberOfFirms)
    # compute the sales
    sales = (identityMatrix - supplyTilde) \ ones(numberOfFirms) / numberOfFirms
    return sales
end

"""
define a function that given an EconomyState it compute the sales of each firm
    - first compute the supplyTilde_ij matrix defined as:
    supplyTilde_ij = (1 - labourIntensity_j) * returnToScale_j * supply_ij
    - then compute the sales as:
        sales = (I - supplyTilde_ij)^(-1) * (1/numberOfFirms)
"""
function computeSales(economyState::EconomyState)
    # get the economy
    economy = economyState.economy
    # get the supply matrix
    supply = economyState.supply
    # get the number of firms
    numberOfFirms = economy.numberOfFirms
    # get the labour intensity vector
    labourIntensity = economy.labourIntensity
    # get the return to scale vector
    returnToScale = economy.returnToScale
    # compute the supplyTilde matrix
    supplyTilde = _computeSupplyTilde(supply, labourIntensity, returnToScale)
    # compute the sales
    sales = _computeSales(supplyTilde, numberOfFirms)
    # return the sales
    return sales
end

"""
define a function that computes wage using the formula:
    wage = sum_j (1 - labourIntensity_j) * returnToScale_j * sales_j * 1/numberOfFirms
"""
function _computeWage(sales, numberOfFirms, labourIntensity, returnToScale)
    wage = 0.0
    @simd for j = 1:numberOfFirms
        wage += (1 - labourIntensity[j]) * returnToScale[j] * sales[j]
    end
    return wage / numberOfFirms
end

"""
define a function that given an EconomyState it compute the wage using the formula:
    wage = sum_j (1 - labourIntensity_j) * returnToScale_j * sales_j * 1/numberOfFirms
"""
function computeWage(economyState::EconomyState)
    # get the economy
    economy = economyState.economy
    # get the sales
    sales = economyState.sales
    # get the number of firms
    numberOfFirms = economy.numberOfFirms
    # get the labour intensity vector
    labourIntensity = economy.labourIntensity
    # get the return to scale vector
    returnToScale = economy.returnToScale
    # compute the wage
    wage = _computeWage(sales, numberOfFirms, labourIntensity, returnToScale)
    # return the wage
    return wage
end

"""
define a function that computes the constant for log prices as:
    constantForLogPrices_i = log(productivity_i^(-1) & returnToScale_i^(- returnToScale_i)
        * sales_i^(1 - returnToScale_i) * wage^(labourIntensity_i * returnToScale_i))
"""
function _computeConstantForLogPricesMainLoop!(
    constantForLogPrices,
    productivity,
    returnToScale,
    sales,
    wage,
    labourIntensity,
    numberOfFirms,
)
    @simd for i = 1:numberOfFirms
        constantForLogPrices[i] = log(
            productivity[i]^(-1) *
            returnToScale[i]^(-returnToScale[i]) *
            sales[i]^(1 - returnToScale[i]) *
            wage^(labourIntensity[i] * returnToScale[i]),
        )
    end
end

function _computeConstantForLogPrices(
    productivity,
    returnToScale,
    sales,
    wage,
    labourIntensity,
)
    # get the number of firms
    numberOfFirms = length(productivity)
    # compute the constant for log prices
    constantForLogPrices = similar(productivity)
    _computeConstantForLogPricesMainLoop!(
        constantForLogPrices,
        productivity,
        returnToScale,
        sales,
        wage,
        labourIntensity,
        numberOfFirms,
    )
    # return the constant for log prices
    return constantForLogPrices
end

"""
define a function that computes the supply prime matrix as:
    supplyPrime_ij = supply_ji * (1 - labourIntensity_i) * returnToScale_i
"""

function _computeSUplyPrimeMainLoop!(supplyPrime, supply, labourIntensity, returnToScale)
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the supplyPrime matrix
    @simd for i = 1:numberOfFirms
        @simd for j = 1:numberOfFirms
            supplyPrime[i, j] = supply[j, i] * (1 - labourIntensity[i]) * returnToScale[i]
        end
    end
end

function _computeSupplyPrime(supply, labourIntensity, returnToScale)
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the supplyPrime matrix
    supplyPrime = similar(supply)
    _computeSUplyPrimeMainLoop!(supplyPrime, supply, labourIntensity, returnToScale)
    # return the supplyPrime matrix
    return supplyPrime
end

"""
define a function that computes the log prices as:
    logPrices = (I - supplyPrime)^(-1) * constantForLogPrices
"""
function _computeLogPrices(supplyPrime, constantForLogPrices)
    # get the number of firms
    numberOfFirms = size(supplyPrime, 1)
    # compute the log prices
    idetityMatrix = Matrix{Float64}(I, numberOfFirms, numberOfFirms)
    logPrices = (idetityMatrix - supplyPrime) \ constantForLogPrices
    # return the log prices
    return logPrices
end

"""
define a function that given an EconomyState it computes the prices
    of goods sold by each firm:
    - first compute the vector constantForLogPrices as:
        constantForLogPrices_i = log(productivity_i^(-1) & returnToScale_i^(- returnToScale_i)
        * sales_i^(1 - returnToScale_i) * wage^(labourIntensity_i * returnToScale_i))
    - second the supplyPrime matrix as:
        supplyPrime_ij = supply_ji * (1 - labourIntensity_i) * returnToScale_i
    - third compute the vector logPrices as:
        logPrices = (I - supplyPrime)^(-1) * constantForLogPrices
    - finally compute the prices as:
        prices = exp(logPrices)
"""
function computePrices(economyState::EconomyState)
    # get the economy
    economy = economyState.economy
    # get the supply matrix
    supply = economyState.supply
    # get the sales
    sales = economyState.sales
    # get the wage
    wage = economyState.wage[1]
    # get the number of firms
    numberOfFirms = economy.numberOfFirms
    # get the labour intensity vector
    labourIntensity = economy.labourIntensity
    # get the return to scale vector
    returnToScale = economy.returnToScale
    # get the productivity vector
    productivity = economy.productivity
    # compute the constantForLogPrices vector
    constantForLogPrices = _computeConstantForLogPrices(
        productivity,
        returnToScale,
        sales,
        wage,
        labourIntensity,
    )
    # compute the supplyPrime matrix
    supplyPrime = _computeSupplyPrime(supply, labourIntensity, returnToScale)
    # compute the logPrices vector
    logPrices = _computeLogPrices(supplyPrime, constantForLogPrices)
    # compute the prices vector
    prices = exp.(logPrices)
    # return the prices
    return prices
end

"""
define a function that computes the employment vector using:
    - employment_i = labourIntensity_i * returnToScale_i * sales_i / wage
"""
function _computeEmploymentMainLoop!(
    employment,
    sales,
    wage,
    labourIntensity,
    returnToScale,
)
    # get the number of firms
    numberOfFirms = length(sales)
    # compute the employment vector
    @simd for i = 1:numberOfFirms
        employment[i] = labourIntensity[i] * returnToScale[i] * sales[i]
    end
end

function _computeEmployment(sales, wage, labourIntensity, returnToScale)
    # get the number of firms
    numberOfFirms = length(sales)
    # compute the employment vector
    employment = similar(sales)
    _computeEmploymentMainLoop!(employment, sales, wage, labourIntensity, returnToScale)
    return employment / wage
end

"""
define a function that given an EconomyState it computes the employment vector using:
    - employment_i = labourIntensity_i * returnToScale_i * sales_i / wage
"""
function computeEmployment(economyState::EconomyState)
    # get the economy
    economy = economyState.economy
    # get the sales
    sales = economyState.sales
    # get the wage
    wage = economyState.wage[1]
    # get the number of firms
    numberOfFirms = economy.numberOfFirms
    # get the labour intensity vector
    labourIntensity = economy.labourIntensity
    # get the return to scale vector
    returnToScale = economy.returnToScale
    # compute the employment vector
    employment = _computeEmployment(sales, wage, labourIntensity, returnToScale)
    # return the employment vector
    return employment
end

"""
define a function that computes the intraFirmsTrade matrix as:
    intraFirmsTrade_ji = (1 - labourIntensity_i) * supply_ij * reuturnToScale_i * sales_i / prices_j
"""
function _computeIntraFirmsTradeMainLoop!(
    intraFirmsTrade,
    supply,
    sales,
    prices,
    labourIntensity,
    returnToScale,
)
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the intraFirmsTrade matrix
    @inbounds @simd for i = 1:numberOfFirms
        for j = 1:numberOfFirms
            @views intraFirmsTrade[i, j] = (1 - labourIntensity[j]) * supply[i, j]
            @views intraFirmsTrade[i, j] *= returnToScale[j] * sales[j] / prices[i]
        end
    end
end

function _computeIntraFirmsTrade(supply, sales, prices, labourIntensity, returnToScale)
    # get the number of firms
    numberOfFirms = size(supply, 1)
    # compute the intraFirmsTrade matrix
    intraFirmsTrade = similar(supply)
    _computeIntraFirmsTradeMainLoop!(
        intraFirmsTrade,
        supply,
        sales,
        prices,
        labourIntensity,
        returnToScale,
    )
    # return the intraFirmsTrade matrix
    return intraFirmsTrade
end

"""
define a function that computes the profits as:
    profits_i = sales_i - employment_i * wage - sum_j prices_j * intraFirmsTrade_ji
"""
function _computeProfitsMainLoop!(profits, sales, employment, wage, prices, intraFirmsTrade)
    # get the number of firms
    numberOfFirms = length(sales)
    # compute the profits vector
    @simd for i = 1:numberOfFirms
        @views profits[i] = sales[i] - employment[i] * wage
        @simd for j = 1:numberOfFirms
            @views profits[i] = profits[i] - prices[j] * intraFirmsTrade[j, i]
        end
    end
end

function _computeProfits(sales, employment, wage, prices, intraFirmsTrade)
    # get the number of firms
    numberOfFirms = length(sales)
    # compute the profits vector
    profits = similar(sales)
    _computeProfitsMainLoop!(profits, sales, employment, wage, prices, intraFirmsTrade)
    # return the profits vector
    return profits
end

"""
define a function that given an EconomyState computes the profits vector using:
    - first computing the matrix intraFirmsTrade as:
        intraFirmsTrade_ji = (1 - labourIntensity_i) * supply_ji * reuturnToScale_i * sales_i / prices_j
    - secodn compute profits as:
        profits_i = sales_i - employment_i * wage - sum_j prices_j * intraFirmsTrade_ji
"""
function computeProfits(economyState::EconomyState)
    # get the economy
    economy = economyState.economy
    # get the supply matrix
    supply = economyState.supply
    # get the sales
    sales = economyState.sales
    # get the prices
    prices = economyState.prices
    # get the wage
    wage = economyState.wage[1]
    # get the labour intensity vector
    labourIntensity = economy.labourIntensity
    # get the return to scale vector
    returnToScale = economy.returnToScale
    # get the employment vector
    employment = economyState.employment
    # compute the intraFirmsTrade matrix
    intraFirmsTrade =
        _computeIntraFirmsTrade(supply, sales, prices, labourIntensity, returnToScale)
    # compute the profits vector
    profits = _computeProfits(sales, employment, wage, prices, intraFirmsTrade)
    # return the profits vector
    return profits
end

"""
define a function that given an EconomyState and a new supply matrix and a firm i
    it computes the expected profits of firm i:
"""
# function computeExpectedProfits(economyState::EconomyState, supply::Matrix{Float64}, i::Int64)
#     # get the economy
#     economy = economyState.economy
#     # get the sales
#     sales = economyState.sales
#     # get the prices
#     prices = economyState.prices
#     # get the wage
#     wage = economyState.wage[1]
#     # get the labour intensity vector
#     labourIntensity = economy.labourIntensity
#     # get the return to scale vector
#     returnToScale = economy.returnToScale
#     # get the employment vector
#     employment = economyState.employment
#     # compute the intraFirmsTrade matrix
#     intraFirmsTrade = _computeIntraFirmsTrade(supply, sales, prices, labourIntensity, 
#                                                                             returnToScale)
#     # compute the profits vector
#     profits = _computeProfits(sales, employment, wage, prices, intraFirmsTrade)
#     println("total profits: ", profits)
#     # return the profits vector
#     return profits[i]
# end

"""
define a function that given an EconomyState it computes the computse the computes
    the equilibrium and stores it in the economyState
"""
function computeEquilibrium!(economyState::EconomyState)
    sales = computeSales(economyState)
    economyState.sales[:] = sales
    wage = computeWage(economyState)
    economyState.wage .= wage
    prices = computePrices(economyState)
    economyState.prices[:] = prices
    employment = computeEmployment(economyState)
    economyState.employment[:] = employment
    profits = computeProfits(economyState)
    economyState.profits[:] = profits
    return nothing
end

"""
define a function that given a supply matrix and a specific firm
    and the firm level of foresight it computes the known firms
    TODO : UGLY CODE - REFACTOR and/or REWRITE
"""
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

"""
define a function that computes the expected sales constant using:
    salesConstant_k = sum_(j in unknownFirms) (1 - labourIntensity_j) * returnToScale_j 
                        * supply_kj * sales_j
    where k is a known firm
"""
function _computePartialExpectedSalesConstantMainLoop!(
    expectedSalesConstant::Vector{Float64},
    knownFirms::Vector{Int64},
    unknownFirms::Vector{Int64},
    supply::Matrix{Float64},
    sales::Vector{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the expected sales constant
    @simd for i = 1:length(knownFirms)
        @simd for j in unknownFirms
            @views expectedSalesConstant[i] +=
                (1 - labourIntensity[j]) *
                returnToScale[j] *
                supply[knownFirms[i], j] *
                sales[j]
        end
    end
    return nothing
end


function _computePartialExpectedSalesConstant(
    knownFirms::Vector{Int64},
    unknownFirms::Vector{Int64},
    supply::Matrix{Float64},
    sales::Vector{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the expected sales constant
    expectedSalesConstant = zeros(length(knownFirms))
    _computePartialExpectedSalesConstantMainLoop!(
        expectedSalesConstant,
        knownFirms,
        unknownFirms,
        supply,
        sales,
        labourIntensity,
        returnToScale,
    )
    return expectedSalesConstant
end

"""
define a function that computes the expected supply tilde using:
    supplyTilde_ij = (1 - labourIntensity_j) * returnToScale_j * supply_ij
    where i and j are known firms
"""
function _computeExpectedSupplyTildeMainLoop!(
    expectedSupplyTilde::Matrix{Float64},
    knownFirms::Vector{Int64},
    supply::Matrix{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the expected supply tilde
    @simd for i = 1:length(knownFirms)
        @simd for j = 1:length(knownFirms)
            expectedSupplyTilde[i, j] =
                (1 - labourIntensity[knownFirms[j]]) *
                returnToScale[knownFirms[j]] *
                supply[knownFirms[i], knownFirms[j]]
        end
    end
    return nothing
end


function _computeExpectedSupplyTilde(
    knownFirms::Vector{Int64},
    supply::Matrix{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the expected supply tilde
    expectedSupplyTilde = zeros(Float64, length(knownFirms), length(knownFirms))
    _computeExpectedSupplyTildeMainLoop!(
        expectedSupplyTilde,
        knownFirms,
        supply,
        labourIntensity,
        returnToScale,
    )
    return expectedSupplyTilde
end

"""
define a function that computes the partialEexpectedSales vector using:
    expectedSales = (I - supplyTilde)^(-1) * (1/numberOfFirms + salesConstant)
"""
function _computePartialExpectedSales(
    knownFirms::Vector{Int64},
    numberOfFirms::Int64,
    expectedSupplyTilde::Matrix{Float64},
    expectedSalesConstant::Vector{Float64},
)
    identityMatrix = Matrix{Float64}(I, length(knownFirms), length(knownFirms))
    expectedSales =
        (identityMatrix - expectedSupplyTilde) \
        (ones(length(knownFirms)) ./ numberOfFirms + expectedSalesConstant)
    return expectedSales
end


function _computeExpectedSales(
    knownFirms::Vector{Int64},
    partialExpectedSales::Vector{Float64},
    sales::Vector{Float64},
)
    # compute the expected sales
    expectedSales = similar(sales)
    count = 1
    for i = 1:length(sales)
        if i in knownFirms
            expectedSales[i] = partialExpectedSales[count]
            count += 1
        else
            expectedSales[i] = sales[i]
        end
    end
    return expectedSales
end

"""
define a function that computes the expected wage by:
    - firts computing the expectedWage constant:
    wageConstant = sum_(k in knownFirms) labourIntensity_k * returnToScale_k * sales_k/ wage#
    - then computing the expected wage as:
    expectedWage = (1/wageConstant) * sum_(k in known firms) labourIntensity_k * returnToScale_k * 
                    expectedSales_k
"""
function _computeExpectedWage(
    knownFirms::Vector{Int64},
    expectedSales::Vector{Float64},
    labourIntensity::Vector{Float64},
    sales::Vector{Float64},
    returnToScale::Vector{Float64},
    wage::Float64,
)
    # compute the expected wage constant
    wageConstant = 0.0
    for i in knownFirms
        wageConstant += labourIntensity[i] * returnToScale[i] * sales[i] / wage
    end
    # compute the expected wage
    # FLAG 1: expected sales is correct?
    expectedWage = 0.0
    for i in knownFirms
        expectedWage += labourIntensity[i] * returnToScale[i] * expectedSales[i]
    end
    expectedWage /= wageConstant
    return expectedWage
end

"""
define a function that computes the first expected log prices constant vector as:
    expectedLogPricesConstant1_k = sum_(i in unknownFirms) (1 - labourIntensity_i) * 
                                    returnToScale_i * supply_ik * logPrice_i
    where k is a known firm
"""

function _computeExpectedLogPricesConstant1(
    knownFirms::Vector{Int64},
    unknownFirms::Vector{Int64},
    supply::Matrix{Float64},
    price::Vector{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the log prices
    logPrice = log.(price)
    # compute the expected log prices constant 1
    expectedLogPricesConstant1 = zeros(Float64, length(knownFirms))
    @simd for i = 1:length(knownFirms)
        @simd for j in unknownFirms
            expectedLogPricesConstant1[i] +=
                (1 - labourIntensity[knownFirms[i]]) *
                returnToScale[knownFirms[i]] *
                supply[j, knownFirms[i]] *
                log(price[j])
        end
    end
    return expectedLogPricesConstant1
end

"""
define a function that computes the second expected log prices constant vector as:
    expectedLogPricesConstant2_k = log(productivity_k^(-1) * returnToScale_k^(- returnToScale_k)
                                        * expectedSales_k^( 1 - returnToScale_k) *
                                        expectedWage_k^(labourIntensity_k * returnToScale_k)) 
                                        + expectedLogPricesConstant1_k
    )
    where k is a known firm
"""
function _computeExpectedLogPricesConstant2(
    knownFirms::Vector{Int64},
    expectedLogPricesConstant1::Vector{Float64},
    expectedSales::Vector{Float64},
    expectedWage::Float64,
    productivity::Vector{Float64},
    returnToScale::Vector{Float64},
    labourIntensity::Vector{Float64},
)
    # compute the expected log prices constant 2
    expectedLogPricesConstant2 = zeros(Float64, length(knownFirms))
    @simd for i = 1:length(knownFirms)
        expectedLogPricesConstant2[i] =
            log(
                productivity[knownFirms[i]]^(-1) *
                returnToScale[knownFirms[i]]^(-returnToScale[knownFirms[i]]) *
                expectedSales[knownFirms[i]]^(1 - returnToScale[knownFirms[i]]) *
                expectedWage^(labourIntensity[knownFirms[i]] * returnToScale[knownFirms[i]]),
            ) + expectedLogPricesConstant1[i]
    end
    return expectedLogPricesConstant2
end

"""
define a functiont that computes the expectedSupplyPrime matrix as:
    expectedSupplyPrime_ij = (1 - labourIntensity_i) * returnToScale_i * supply_ji
    where i and j are known firms
"""
function _computeExpectedSupplyPrimeMainLoop!(
    expectedSupplyPrime::Matrix{Float64},
    knownFirms::Vector{Int64},
    supply::Matrix{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    @simd for i = 1:length(knownFirms)
        @simd for j = 1:length(knownFirms)
            expectedSupplyPrime[j, i] =
                (1 - labourIntensity[knownFirms[i]]) *
                returnToScale[knownFirms[i]] *
                supply[knownFirms[j], knownFirms[i]]
        end
    end
    return expectedSupplyPrime
end


function _computeExpectedSupplyPrime(
    knownFirms::Vector{Int64},
    supply::Matrix{Float64},
    labourIntensity::Vector{Float64},
    returnToScale::Vector{Float64},
)
    # compute the expected supply prime
    expectedSupplyPrime = zeros(Float64, length(knownFirms), length(knownFirms))
    _computeExpectedSupplyPrimeMainLoop!(
        expectedSupplyPrime,
        knownFirms,
        supply,
        labourIntensity,
        returnToScale,
    )
    return expectedSupplyPrime
end

"""
define a function to compute the partialExpectedPrices as a vector of length numberOfFirms
    logpartialExpectedPrices = (I - expectedSupplyPrime)^(-1) * expectedLogPricesConstant2
    partialExpectedPrices = exp.(logpartialExpectedPrices)
"""
function _computePartialExpectedPrices(
    knownFirms::Vector{Int64},
    expectedSupplyPrime::Matrix{Float64},
    expectedLogPricesConstant2::Vector{Float64},
)
    # compute the partial expected prices
    identityMatrix = Matrix{Float64}(I, length(knownFirms), length(knownFirms))
    logpartialExpectedPrices =
        (identityMatrix - expectedSupplyPrime) \ expectedLogPricesConstant2
    partialExpectedPrices = exp.(logpartialExpectedPrices)
    return partialExpectedPrices
end

"""
define a function that computes the expected prices as a vector of length numberOfFirms
    expectedPrices = zeros(numberOfFirms)
    for i in 1:numberOfFirms
        if i in knownFirms
            expectedPrices[i] = partialExpectedPrices[i]
        else
            expectedPrices[i] = price[i]
        end
    end
"""
function _computeExpectedPrices(
    knownFirms::Vector{Int64},
    partialExpectedPrices::Vector{Float64},
    price::Vector{Float64},
)
    # compute the expected prices
    expectedPrices = similar(price)
    count = 1
    @simd for i = 1:length(price)
        if i in knownFirms
            expectedPrices[i] = partialExpectedPrices[count]
            count += 1
        else
            expectedPrices[i] = price[i]
        end
    end
    return expectedPrices
end

"""   
define a function that given an EconomyState in equilibrium and a new supply matrix
    and a firm i it computes the expected profits of firm i
"""
function computeExpectedProfits(
    economyState::EconomyState,
    newSupply::Matrix{Float64},
    firm::Int64,
)
    # get the economy
    economy = economyState.economy
    # get the current supply matrix
    supply = economyState.supply
    # get the number of firms
    NumberOfFirms = economy.numberOfFirms
    # get the foresight for firm
    foresight = economy.foresight[firm]
    # get the known firms
    knownFirms = _computeKnownFirms(firm, foresight, supply)
    # get the unknown firms
    unknownFirms = _computeUnknownFirms(NumberOfFirms, knownFirms)
    # compute the expectedSales constant
    expectedSalesConstant = _computePartialExpectedSalesConstant(
        knownFirms,
        unknownFirms,
        newSupply,
        economyState.sales,
        economy.labourIntensity,
        economy.returnToScale,
    )
    # compute hte expectedSupplyTilde matrix
    expectedSupplyTilde = _computeExpectedSupplyTilde(
        knownFirms,
        newSupply,
        economy.labourIntensity,
        economy.returnToScale,
    )
    # compute the expectedSales
    # (knownFirms::Vector{Int64}, partialExpectedSales::Vector{Float64}, sales::Vector{Float64})
    partialExpectedSales = _computePartialExpectedSales(
        knownFirms,
        NumberOfFirms,
        expectedSupplyTilde,
        expectedSalesConstant,
    )
    expectedSales =
        _computeExpectedSales(knownFirms, partialExpectedSales, economyState.sales)
    # compute the expectedWage
    expectedWage = _computeExpectedWage(
        knownFirms,
        expectedSales,
        economy.labourIntensity,
        economyState.sales,
        economy.returnToScale,
        economyState.wage[1],
    )

    # compute the expectedLogPricesConstant1
    expectedLogPricesConstant1 = _computeExpectedLogPricesConstant1(
        knownFirms,
        unknownFirms,
        newSupply,
        economyState.prices,
        economy.labourIntensity,
        economy.returnToScale,
    )
    # compute the expectedLogPricesConstant2
    expectedLogPricesConstant2 = _computeExpectedLogPricesConstant2(
        knownFirms,
        expectedLogPricesConstant1,
        expectedSales,
        expectedWage,
        economy.productivity,
        economy.returnToScale,
        economy.labourIntensity,
    )

    # compute the expected supplyPrime matrix
    expectedSupplyPrime = _computeExpectedSupplyPrime(
        knownFirms,
        supply,
        economy.labourIntensity,
        economy.returnToScale,
    )

    # compute the partial expectedPrices
    partialExpectedPrices = _computePartialExpectedPrices(
        knownFirms,
        expectedSupplyPrime,
        expectedLogPricesConstant2,
    )
    # compute the expectedPrices
    expectedPrices =
        _computeExpectedPrices(knownFirms, partialExpectedPrices, economyState.prices)
    # compute the expectedEmployment
    expectedEmployment = _computeEmployment(
        expectedSales,
        expectedWage,
        economy.labourIntensity,
        economy.returnToScale,
    )
    # compute the expectedIntraFirmsTrade
    expectedIntraFirmsTrade = _computeIntraFirmsTrade(
        newSupply,
        expectedSales,
        expectedPrices,
        economy.labourIntensity,
        economy.returnToScale,
    )
    # compute the expectedProfits
    expectedProfits = _computeProfits(
        expectedSales,
        expectedEmployment,
        expectedWage,
        expectedPrices,
        expectedIntraFirmsTrade,
    )
    return expectedProfits[firm]
end
