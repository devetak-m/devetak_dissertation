using Random

function getParameters()
    return [0.005, 2.0, 50.0, 0.5, 2.0, 1.0, 0.1, 0.1, 0.5, 0.02, 2.0, 0.5, 0.1, 0.2]
    # rho0, theta, Gamma0, C0,  R,  r,  gammap, eta0m, bankrupcyEffect, delta, beta, tau_meas, phi, taupi
end

"""

    mark0_noCB(par, N, seed, maxIter)

    Implements mark0 with no central bank. 
    
    Assumes initial porduction(y0) is 0.5
    Assumes Gammas is 0.0
    Assumes real rate effect on consumption(alpha) is 4.0 <- does Karl do this?
    Assumes factor to adjust wages to inflation expectations is 1.0

"""
function mark0_noCB(par, N, seed, maxIter, cutOff = 0)

    # set seed, ignore for AD
    Random.seed!(seed)

    # unpack parameters vector
    rho0 = par[1]
    theta = par[2] # grad = 0 beacuse no calculations, how to include?
    Gamma0 = par[3]
    C0 = par[4] # assumed between 0 and 1
    R = par[5]
    r = par[6]
    gammap = par[7]
    eta0m = par[8]
    bankrupcyEffect = par[9] # grad = 0, why?!
    delta = par[10]
    beta = par[11]
    tau_meas = par[12]
    phi = par[13] # grad = 0 as expected because no StochasticAD.jl
    taupi = par[14]

    # generate parameter variables
    ao = one(typeof(C0))
    M0 = ao * N
    eta0p = R * eta0m
    gammaw = r * gammap
    Pavg = one(typeof(C0))
    Pold = one(typeof(Pavg))
    inflation = pi_avg = zero(typeof(theta))
    rhom = rho = rm_avg = rho0

    rhop = rp_avg = zero(typeof(rm_avg))
    u_avg = zero(typeof(rm_avg))

    # initialize big arrays
    # already correct
    alive = ones(Bool, N)
    demand = ones(typeof(C0), N) * 0.5
    wage = ones(typeof(C0), N)

    # to compute
    price = zeros(typeof(C0), N)
    production = zeros(typeof(C0), N)
    profit = zeros(typeof(C0), N)
    assets = zeros(typeof(C0), N)
    # smaller variables
    Wtot = zero(typeof(C0))
    Ytot = zero(typeof(C0))
    tmp = zero(typeof(C0))

    @inbounds for i = 1:N
        price[i] = one(typeof(C0)) + (0.02) * rand(typeof(C0)) - 0.01
        production[i] = 0.5 + (0.02) * rand(typeof(C0)) - 0.01
        profit[i] = price[i] * min(production[i], demand[i]) - wage[i] * production[i]
        assets[i] = 2 * production[i] * wage[i] * rand(typeof(C0))
        Wtot += wage[i] * production[i]
        Ytot += production[i]
        tmp += price[i] * production[i]
    end

    # initialize smaller stuff
    e = sum(production) / N
    u = 1 - e
    # correct this
    Pavg = tmp / Ytot
    Wavg = Wtot / Ytot
    Wmax = one(typeof(Wavg))
    S = e * N
    bust = 0
    revived = zeros(maxIter)


    # fix total ammount of money to N
    tmp = sum(assets) + S
    S = S * N / tmp * ao

    @. assets = assets * N / tmp * ao

    # replace with time-series
    unemploymentTimeSeries = zeros(typeof(u), maxIter - cutOff)

    Wtot = zero(typeof(Wavg))
    Ytot = zero(typeof(Wavg))
    tmp = zero(typeof(Wavg))
    Pmin = Inf

    rp = zeros(typeof(Wavg), N)
    rw = zeros(typeof(Wavg), N)
    payRoll = zeros(typeof(Wavg), N)

    # initialize in loop va

    # main loop
    @inbounds for turn = 1:maxIter

        # renorm step
        @. price = price / Pavg
        @. wage = wage / Pavg
        @. assets = assets / Pavg
        @. profit = profit / Pavg

        # old
        S /= Pavg
        Wavg /= Pavg
        Wmax /= Pavg
        Pold /= Pavg
        M0 /= Pavg
        Pavg = one(typeof(Pavg))

        # small updates
        pi_avg = taupi * inflation + (1 - taupi) * pi_avg
        rp_avg = taupi * rhop + (1 - taupi) * rp_avg
        rm_avg = taupi * rhom + (1 - taupi) * rm_avg
        u_avg = taupi * u + (1 - taupi) * u_avg

        wageNorm = zero(typeof(Wmax))
        if beta > 0
            for i in eachindex(alive)
                if alive[i]
                    wageNorm += exp(beta * (wage[i] .- Wmax) / Wavg)
                end
            end
        end

        pi_used = tau_meas * pi_avg

        Gamma = max(Gamma0 * (rm_avg - pi_used), zero(typeof(Gamma0)))

        deftot = zero(typeof(assets[1]))
        tmp = zero(typeof(tmp))
        Wtot = zero(typeof(Wtot))
        firm_savings = zero(typeof(assets[1]))
        Ytot = zero(typeof(Ytot))
        debt_tot = zero(typeof(assets[1]))
        Pmin = Inf

        @inbounds for i = 1:N
            if alive[i]
                payRoll = production[i] * wage[i]
                if assets[i] > -theta * payRoll # theta > 0 assumed
                    if payRoll > zero(typeof(payRoll))
                        ren = Gamma * assets[i] / payRoll
                    else
                        ren = zero(typeof(Gamma))
                    end

                    rp = gammap * rand()
                    rw = gammaw * rand()

                    excessDemand = demand[i] - production[i]

                    if beta > zero(typeof(beta))
                        arg = beta * (wage[i] - Wmax) / Wavg
                        u_share = u * N * (1 - bust) * exp(arg) / wageNorm
                    else
                        u_share = u
                    end

                    if excessDemand > 0
                        eta = eta0p * (1 + ren)
                        if eta < zero(typeof(eta))
                            eta = zero(typeof(eta))
                        elseif eta > one(typeof(eta))
                            eta = one(typeof(eta))
                        end

                        production[i] += min(eta * excessDemand, u_share)

                        if price[i] < Pavg
                            price[i] *= (one(typeof(rp)) + rp)
                        end

                        if profit[i] > 0
                            wage[i] *= one(typeof(ren)) + (one(typeof(ren)) + ren) * rw * e
                            wage[i] = min(
                                wage[i],
                                (
                                    price[i] * min(demand[i], production[i]) +
                                    rhom * min(assets[i], zero(typeof(ren))) +
                                    rhop * max(assets[i], zero(typeof(ren)))
                                ) / production[i],
                            )
                            wage[i] = max(wage[i], zero(typeof(wage[i])))
                        end
                    else
                        eta = eta0m * (one(typeof(ren)) - ren)
                        if eta < zero(typeof(eta))
                            eta = zero(typeof(eta))
                        elseif eta > one(typeof(eta))
                            eta = one(typeof(eta))
                        end

                        production[i] += eta * excessDemand

                        if price[i] > Pavg
                            price[i] *= (one(typeof(rp)) - rp)
                        end

                        if profit[i] < zero(typeof(profit[i]))
                            wage[i] *=
                                one(typeof(Wavg)) - (one(typeof(Wmax)) - ren) * rw * u
                            wage[i] = max(wage[i], zero(typeof(Wavg)))
                        end
                    end

                    # anticipate inflation
                    price[i] *= one(Pavg) + pi_used
                    wage[i] *= one(Wavg) + pi_used

                    production[i] = max(production[i], zero(typeof(production[i])))

                    Wtot += production[i] * wage[i]
                    tmp += price[i] * production[i]
                    firm_savings += max(assets[i], zero(typeof(assets[1])))
                    debt_tot -= min(assets[i], zero(typeof(assets[1])))
                    Ytot += production[i]

                    Pmin = min(Pmin, price[i])

                else # firm bankrupted
                    alive[i] = false
                    deftot -= assets[i]
                end
            end
        end

        # update statistics
        Wavg = Wtot / Ytot
        Pavg = tmp / Ytot
        e = Ytot / N
        u = 1 - e

        # correct for numerical errors
        if abs(S + firm_savings - debt_tot - deftot - M0) > 0
            left = S + firm_savings - debt_tot - deftot - M0
            S -= left
        end

        rhom = rho
        if debt_tot > 0
            rhom += (1 - bankrupcyEffect) * deftot / debt_tot
        end

        interests = rhom * debt_tot

        rhop = zero(typeof(rhom))
        k = zero(typeof(rhom))

        if S + firm_savings > 0
            rhop = (interests - deftot) / (S + firm_savings)
            k = debt_tot / (S + firm_savings)
        end

        S += rhop * S

        # consumption

        # test:
        G = C0 * (1 + 4.0 * (pi_used - rp_avg))
        if G > 1
            G = 1
        elseif G < 0
            G = 0
        end
        budget = G * (Wtot + max(S, zero(typeof(S))))

        # budget = C0 * (Wtot + max(S, 0)) 

        Pnorm = zero(typeof(Pmin))
        @inbounds for i = 1:N
            if alive[i]
                Pnorm += exp((Pmin - price[i]) * beta / Pavg)
            end
        end

        firm_savings = zero(typeof(assets[1]))

        @inbounds for i = 1:N
            if alive[i]
                arg = beta * (Pmin - price[i]) / Pavg
                demand[i] = budget * exp(arg) / (Pnorm * price[i])
                precalculate =
                    price[i] * min(production[i], demand[i]) - production[i] * wage[i]
                # interestPayments = assets[i] < zero(typeof(assets[1])) ? rhom * assets[i] : rhop * assets[i] <- LLVM figures out this?!
                profit[i] =
                    precalculate +
                    rhom * min(assets[i], zero(typeof(assets[1]))) +
                    rhop * max(assets[i], zero(typeof(assets[1])))

                S -= precalculate
                assets[i] += profit[i]

                if assets[i] > zero(typeof(assets[1]))
                    firm_savings += assets[i]
                    if profit[i] > zero(typeof(profit[1]))
                        S += delta * assets[i]
                        assets[i] *= (one(typeof(delta)) - delta)
                    end
                end
            end
        end

        # revival turn
        deftot = zero(typeof(deftot))
        for i in eachindex(alive)
            if !alive[i] && rand() < phi
                alive[i] = true
                production[i] = u * rand() # max.(u, 0) * rand()
                price[i] = Pavg
                wage[i] = Wavg
                assets[i] = Wavg * production[i]
                firm_savings += assets[i]
                profit[i] = zero(typeof(profit[1]))
                deftot += assets[i]
            end
        end

        # unemploymentTest = 1 - sum(production)/N
        # unemploymentTest < 0 && @show unemploymentTest

        tmp = zero(typeof(tmp))
        Ytot = zero(typeof(Ytot))
        Wtot = zero(typeof(Wtot))
        bust = zero(typeof(bust))
        debt_tot = zero(typeof(debt_tot))
        Wmax = zero(typeof(Wmax))

        # final
        @inbounds for i = 1:N
            if alive[i]
                if firm_savings > zero(typeof(firm_savings)) &&
                   assets[i] > zero(typeof(assets[1]))
                    assets[i] -= deftot * assets[i] / firm_savings
                end

                Wtot += production[i] * wage[i]
                Ytot += production[i]
                tmp += price[i] * production[i]
                debt_tot -= min(assets[i], zero(typeof(debt_tot)))

                if wage[i] > Wmax
                    Wmax = wage[i]
                end
            else
                bust += one(typeof(bust)) / N
            end
        end

        # update statistics
        Pavg = tmp / Ytot
        Wavg = Wtot / Ytot

        inflation = (Pavg - Pold) / Pold
        Pold = Pavg
        e = Ytot / N
        # e = min(e, one(typeof(e)))
        # e = max(e, zero(typeof(e)))
        u = 1 - e
        u < 0 && @show u

        if turn > cutOff
            unemploymentTimeSeries[turn-cutOff] = Ytot
        end

    end

    @show sum(revived .> 5)

    return unemploymentTimeSeries

end


# mark0_noCB(getParameters(), 2000, 1, 100)
