function model()
    
    # Here's the model - nice + simple :)
    purpose = Dict(1 => :distillation, 2 => :other)
    pa = Dict(1 => :present, 2 => :absent)

    lab_prior = [0.2, 0.8]
    lab_purpose = DiscreteRoot(lab_prior)(:lab_purpose)

    # Capabilities.
    condensing = DiscreteCPD([[0.99, 0.01], [0.05, 0.95]])(:condensing)
    distill_vessel = DiscreteCPD([[0.9, 0.1], [0.05, 0.95]])(:distill_vessel)
    collection_vessel = DiscreteCPD([[0.6, 0.4], [0.45, 0.55]])(:collection_vessel)
    heating = DiscreteCPD([[0.6, 0.4], [0.45, 0.55]])(:heating)
    temp = DiscreteCPD([[0.6, 0.4], [0.45, 0.55]])(:temp)
    separation = DiscreteCPD([[0.3, 0.7], [0.8, 0.2]])(:separation)

    # Implements.
    condenser = DiscreteCPD([[0.9, 0.1],
                             [0.05, 0.95]])(:condenser)

    three_neck_flask = DiscreteCPD([[0.5, 0.5],
                                    [0.05, 0.95]])(:three_neck_flask)

    round_bottom_flask = DiscreteCPD([[0.99, 0.01],
                                      [0.8, 0.2],
                                      [0.8, 0.2],
                                      [0.05, 0.95]]
                                    )(:round_neck_flask)

    erlenmeyer_flask = DiscreteCPD([[0.9, 0.1],
                                    [0.05, 0.95]])(:erlenmeyer_flask)

    beaker = DiscreteCPD([[0.9, 0.1],
                          [0.05, 0.95]])(:beaker)

    hot_plate = DiscreteCPD([[0.9, 0.1],
                             [0.05, 0.95]])(:hot_plate)

    heating_mantel = DiscreteCPD([[0.9, 0.1],
                                  [0.05, 0.95]])(:heating_mantel)

    bunsen = DiscreteCPD([[0.9, 0.1],
                          [0.05, 0.95]])(:bunsen)

    digital_thermo = DiscreteCPD([[0.9, 0.1],
                                  [0.05, 0.95]])(:digital_thermometer)

    mercury_thermo = DiscreteCPD([[0.9, 0.1],
                                  [0.2, 0.8]])(:mercury_thermometer)

    separatory_funnel = DiscreteCPD([[0.9, 0.1],
                                     [0.2, 0.8]])(:separatory_funnel)

    # Network + runtime.
    smile_network = Network(Tuple{}, Nothing)
    add_variable!(smile_network, lab_purpose)

    # Capabilities.
    add_variable!(smile_network, condensing, [lab_purpose])
    add_variable!(smile_network, distill_vessel, [lab_purpose])
    add_variable!(smile_network, collection_vessel, [lab_purpose])
    add_variable!(smile_network, heating, [lab_purpose])
    add_variable!(smile_network, temp, [lab_purpose])
    add_variable!(smile_network, separation, [lab_purpose])

    # Implements.
    add_variable!(smile_network, condenser, [condensing])
    add_variable!(smile_network, three_neck_flask, [distill_vessel])
    add_variable!(smile_network, round_bottom_flask, [distill_vessel, collection_vessel])
    add_variable!(smile_network, erlenmeyer_flask, [collection_vessel])
    add_variable!(smile_network, beaker, [collection_vessel])
    add_variable!(smile_network, hot_plate, [heating])
    add_variable!(smile_network, heating_mantel, [heating])
    add_variable!(smile_network, bunsen, [heating])
    add_variable!(smile_network, digital_thermo, [temp])
    add_variable!(smile_network, mercury_thermo, [temp])
    add_variable!(smile_network, separatory_funnel, [separation])

    runtime = Runtime(smile_network)

    # Initialize runtime for BP.
    default_initializer(runtime)
    return runtime
end