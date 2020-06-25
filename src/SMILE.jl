module SMILE

using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.Operators
using Scruff.Algorithms
using TextParse

# Parsing.
using CSV
using DataFrames

include("model.jl")

function run(args, tol = 0.7)
    isempty(args) && error("CommandLineArgsError: you haven't provided an input file path.")

    # Parsing.
    df = CSV.read(pwd() * "/" * args[1]; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true)
    label_dict = Dict(0 => :beaker, 
                      1 => :separatory_funnel,
                      2 => :digital_thermometer,
                      3 => :erlenmeyer_flask,
                      4 => :three_neck_flask,
                      5 => :condenser,
                      6 => :heating_mantel,
                      7 => :hot_plate,
                      8 => :bunsen_burner,
                      9 => :round_neck_flask,
                      10 => :mercury_thermometer)

    # Toplevel purpose.
    purpose = Dict(1 => :distillation, 2 => :other)

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

    runtime = SMILE.model()

    # Get current instances.
    instances = Dict(map(get_variables(runtime.network)) do v
                         get_name(v) => current_instance(runtime, v)
                     end)
    collected = collect(instances)

    # Get prior beliefs.
    text = map(collected) do (k, v)
        "$(k) : $(get_params(v.var.definition))\n"
    end
    @info "Priors.\n$(text...)"

    # Set initial soft evidence (beliefs) from neural network output.
    for (k, v) in priors
        l = label_dict[k]
        @info "Posting soft evidence to $l. Probability of presence is $v."
        set_value!(runtime, instances[l], :evidence, Dict([1 => v, 2 => 1.0 - v]))
    end

    # Perform BP.
    loopybp(runtime)

    # Get updated beliefs for lab capability.
    belief = get_value(runtime, instances[:lab_purpose], :belief) 

    # Set decision over what sort of lab it is.
    max_index = findall(a -> a .== maximum(belief), belief)[1]
    max = purpose[max_index]
    @info "Setting decision...$(max). Propagating back down the network."
    set_value!(runtime, instances[:lab_purpose], :evidence, max_index)

    # Propagate back down with BP.
    Scruff.Algorithms.three_pass_BP(runtime)

    # Get new beliefs.
    text = map(collected) do (k, v)
        "$(k) : $(get_value(runtime, v, :belief))\n"
    end
    @info "Posterior marginal beliefs.\n$(text...)"
    likely_present = String[]
    for (k, v) in collected
        bel = get_value(runtime, v, :belief)
        if bel[1] > tol
            push!(likely_present, "$(k) : $(bel[1])\n")
        end
    end
    @info "Probability of presence greater than $tol.\n$(likely_present...)"
end

run(ARGS)

end # module

