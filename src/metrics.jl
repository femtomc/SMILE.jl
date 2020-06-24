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

# Required for efficient unzipping of Array{Tuple}.
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

include("model.jl")

function object_detection_recall(label_dir::String, validation_dir::String)
    label_files = readdir(label_dir, join = true)
    predict_files = readdir(validation_dir, join = true)
    total = 0.0
    recall = 0.0
    incorrect = Tuple{String, String}[]
    for (pl, l) in zip(predict_files, label_files)
        # Get ground truth.
        gt_labels = map(eachrow(CSV.read(l; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true))) do k
            k[:Label]
        end

        # Parsing.
        pred_labels = map(eachrow(CSV.read(pl; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true))) do k
            k[:Label]
        end
        for (i, j) in zip(gt_labels, pred_labels)
            if i == j
                recall += 1.0
                total += 1.0
            else
                push!(incorrect, (pl, l))
                total += 1.0
            end
        end
    end
    recall /= total
    recall, collect(Set(incorrect))
end

function object_detection_recall(label_files::Vector{String}, predict_files::Vector{String})
    total = 0.0
    recall = 0.0
    println("Recognition model misses on files:")
    for (pl, l) in zip(predict_files, label_files)
        # Get ground truth.
        gt_labels = map(eachrow(CSV.read(l; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true))) do k
            k[:Label]
        end

        # Parsing.
        pred_labels = map(eachrow(CSV.read(pl; header=["Label", "Probability"], datarow=1, delim = ' ', silencewarnings=true))) do k
            k[:Label]
        end
        len = length(gt_labels)
        local_total = 0.0
        for (i, j) in zip(gt_labels, pred_labels)
            if i == j
                recall += 1.0
                total += 1.0
                local_total += 1.0
            else
                total += 1.0
            end
        end
        println("$len : $(local_total / len) => $pl")
    end
    recall /= total
    recall
end

function model_recall(label_dir::String, validation_dir::String)
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
        for (k, v) in collected
            belief = get_value(runtime, v, :belief)
            if haskey(num_dict, k)
                if belief[1] > 0.8 && num_dict[k] in gt_set
                    success += 1.0/length(gt_set)
                end
            end
        end
        push!(recalls, success)
    end
    sum(recalls) / length(recalls)
end

function model_recall(label_files::Vector{String}, predict_files::Vector{String})

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
                if belief[1] > 0.8 && num_dict[k] in gt_set
                    success += 1.0/length(gt_set)
                end
            end
        end
        push!(recalls, success)
    end
    sum(recalls) / length(recalls)
end

# Compute.
recall, zipped = object_detection_recall(ARGS[1], ARGS[2])
gt, pred = unzip(zipped)
println("Detector recall: $(recall)")
println("Model total 'presence' recall: $(model_recall(ARGS[1], ARGS[2]))")
println("Detector recall on incorrect: $(object_detection_recall(gt, pred))")
println("Model 'presence' on incorrect: $(model_recall(gt, pred))")

end # module
