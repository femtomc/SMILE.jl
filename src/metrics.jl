module Metrics

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

# Setup metric dir (or don't do anything, if already exists).
metric_dir = "metrics"
if !(metric_dir in readdir(pwd()))
    mkdir(metric_dir)
end

function generate_metrics(label_dir, validation_dir)
    label_files = readdir(label_dir, join = true)
    predict_files = readdir(validation_dir, join = true)

    # Model.
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
    num_dict = Dict(map(reverse, collect(label_dict)))
    purpose = Dict(1 => :distillation, 2 => :other)

    recalls = Float64[]

    open("recall.txt", "w") do f
        for (pl, l) in zip(predict_files, label_files)
           
            # Instantiate.
            runtime = model()

            # Get current instances.
            instances = Dict(map(get_variables(runtime.network)) do v
                                 get_name(v) => current_instance(runtime, v)
                             end)

            collected = collect(instances)

            # Get ground truth.
            gt_labels = map(eachrow(CSV.read(l; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true))) do k
                k[:Label]
            end

            # Parsing.
            df = CSV.read(pl; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true)

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

            # Set initial soft evidence (beliefs).
            for (k, v) in priors
                l = label_dict[k]
                set_value!(runtime, instances[l], :evidence, Dict(1 => v, 2 => 1.0 - v))
            end

            # Perform BP.
            loopybp(runtime)

            # Get updated beliefs for lab capability.
            belief = get_value(runtime, instances[:lab_purpose], :belief) 

            # Set decision over what sort of lab it is.
            max_index = findall(a -> a .== maximum(belief), belief)[1]
            max = purpose[max_index]
            set_value!(runtime, instances[:lab_purpose], :evidence, max_index)

            # Propagate back down with BP.
            Scruff.Algorithms.three_pass_BP(runtime)

            # Collect metrics.
            gt_set = Set(gt_labels)
            success = 0.0
            beliefs = String[]
            for (k, v) in collected
                belief = get_value(runtime, v, :belief)
                push!(beliefs, "$(k) : $(belief)\n")
                if haskey(num_dict, k)
                    if belief[1] > 0.5 && num_dict[k] in gt_set
                        success += 1.0/length(gt_set)
                    end
                end
            end
            push!(recalls, success)
            write(f, "$(pl) : $(success)\n$(beliefs...)\n---")

        end
    end
    println(sum(recalls) / length(recalls))
end

generate_metrics(ARGS[1], ARGS[2])
end # module
