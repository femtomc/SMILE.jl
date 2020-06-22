module Parser

using CSV
using DataFrames

# Parsing.
df = CSV.read(pwd() * "/apps/nn_out.txt"; header=["Label", "Probability"], datarow=1, delim = ' ')

# Get priors.
priors = Dict{Int, Float64}()
map(eachrow(df)) do k
    l = k[:Label]
    p = k[:Probability]
    if haskey(priors, l)
        priors[l] < p && begin
            priors[l] = p
        end
    else
        priors[l] = p
    end
end
println(priors)

end #module
