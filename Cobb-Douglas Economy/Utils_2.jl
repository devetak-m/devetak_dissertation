function getNewSupplyMatrix(technologMatrix)
    # create a copy of the technologMatrix
    supplyMatrix = zeros(size(technologMatrix))

    # loop over the columns of the technologMatrix
    for j in 1:size(technologMatrix, 2)
        # get the non-zero indices of the column
        nonZeroIndices = findall(!iszero, technologMatrix[:, j])
        
        # select random non-zero indices
        selectedIndices = rand(nonZeroIndices)
        
        # compute the sum of the selected entries
        selectedSum = technologMatrix[selectedIndices, j]
        
        # add additional random non-zero entries until the sum is one
        while selectedSum < 1
            # get the remaining non-zero indices
            remainingIndices = setdiff(nonZeroIndices, selectedIndices)
            
            # select a random non-zero index
            additionalIndex = rand(remainingIndices)
            
            # compute the sum of the selected entries with the additional entry
            additionalSum = selectedSum + technologMatrix[additionalIndex, j]
            
            # if the sum is less than or equal to one, add the additional entry
            if additionalSum <= 1
                selectedIndices = vcat(selectedIndices, additionalIndex)
                selectedSum = additionalSum
            else
                break
            end
        end

        supplyMatrix[selectedIndices, j] = technologMatrix[selectedIndices, j]
    end
    return supplyMatrix
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

function getHetEconomy(numberOfFirms, technologyMatrix, supplyMatrix, numberOfRounds, t)

    labourIntensity = ones(numberOfFirms) .* 0.3
    returnToScale = ones(numberOfFirms) .* 0.9

    productivity = rand(numberOfFirms) * 3.0

    foresight = ones(numberOfFirms) * t

    economy = Economy(
        numberOfFirms,
        labourIntensity,
        productivity,
        returnToScale,
        foresight,
        technologyMatrix,
    )

    profits = zeros(numberOfFirms)
    prices = zeros(numberOfFirms)
    sales = zeros(numberOfFirms)
    wages = zeros(1)
    employment = zeros(numberOfFirms)

    economyState =
        EconomyState(economy, supplyMatrix, profits, prices, sales, wages, employment)

    dynamicsTimeSeries = DynamicsTimeSeries(economyState, numberOfRounds)

    computeEquilibrium!(dynamicsTimeSeries.economyState)

    return dynamicsTimeSeries

end

function getHomEconomy(numberOfFirms, technologyMatrix, supplyMatrix, numberOfRounds, t)

    labourIntensity = ones(numberOfFirms) .* 0.3
    returnToScale = ones(numberOfFirms) .* 0.9

    productivity = ones(numberOfFirms) * 1.5

    foresight = ones(numberOfFirms) * t

    economy = Economy(
        numberOfFirms,
        labourIntensity,
        productivity,
        returnToScale,
        foresight,
        technologyMatrix,
    )

    profits = zeros(numberOfFirms)
    prices = zeros(numberOfFirms)
    sales = zeros(numberOfFirms)
    wages = zeros(1)
    employment = zeros(numberOfFirms)

    economyState =
        EconomyState(economy, supplyMatrix, profits, prices, sales, wages, employment)

    dynamicsTimeSeries = DynamicsTimeSeries(economyState, numberOfRounds)

    computeEquilibrium!(dynamicsTimeSeries.economyState)

    return dynamicsTimeSeries

end