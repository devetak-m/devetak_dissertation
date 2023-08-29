abstract type DynamicsType end

struct RandomDynamics <: DynamicsType
    prob::Float64
end

struct DeterministicDynamics <: DynamicsType
    tol::Float64
end
