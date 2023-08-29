
function constructHessian!(hessian, jacobian, outputsSquared)
    # jacobian is T × P × S
    # where T - timesteps, P - parameters, S - seeds
    S = size(jacobian, 3)
    T = size(jacobian, 1)
    @inbounds for i in axes(hessian, 1)
        for j in axes(hessian, 2)
            for s = 1:S
                for t = 1:T
                    hessian[i, j] +=
                        jacobian[t, i, s] * jacobian[t, j, s] / outputsSquared[t, s]
                end
            end
            hessian[i, j] /= S * T
        end
    end
    return nothing
end

function constructHessian(jacobian, output)
    P = size(jacobian, 2)
    hessian = zeros(P, P)
    outputSquared = output .^ 2
    constructHessian!(hessian, jacobian, outputSquared)
    return hessian
end
