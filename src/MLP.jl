using Flux, Metal, MLUtils, Statistics, ProgressMeter

using MLDatasets: CIFAR10

# Get the 60,000 32x32 pixel color image data
function cifar10_data() 
    # Split the 60,000 images into training and testing observations
    # and make sure we have normalized Float32 pixel data.
    (CIFAR10(Tx=Float32, split=:train), CIFAR10(Tx=Float32, split=:test))
end

function onehotlabels(data ::CIFAR10)
    onehotbatch(data.targets, range(extrema(data.targets)...))
end

function loader(data::CIFAR10; batchsize)
    x = data.features
    y = onehotlabels(data)
    Flux.DataLoader((x, y); batchsize, shuffle=true)
end

function accuracy(model, data::CIFAR10, args::TrainingParams)
    (x, y) = only(loader(data; batchsize=(length(data))))
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)
    round(100 * mean(iscorrect); digits=2)
end

struct TrainingParams
    batchsize :: Int
    epochs    :: Int
    learnrate :: Float64
end

# constructor with keyword args
function trainparams(;batchsize::Int, epochs::Int, learnrate::Float64)
    TrainingParams(batchsize, epochs, learnrate)
end

# TODO paramterise: batchsize, epochs, training rate
function trainwith(model, train_data::CIFAR10, args::TrainingParams)
    @info "trainwith" args
    # loader
    train_loader = loader(train_data, batchsize=args.batchsize)
    # optimizer state with training rate
    opt_state = Flux.setup(Adam(args.learnrate), model)
    losses = [] # keep track of loss at each epoch
    
    @showprogress for epoch in 1:args.epochs
        for (x, y) in train_loader
            # compute loss and gradients
            l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
            # update model parameters
            Flux.update!(opt_state, model, gs[1]) # see: withgradient
            # accumulate losses for logging
            push!(losses, l)
        end
    end
    return (model, losses, length(train_loader))
end

using Plots

function trainandtest(model, tparam::TrainingParams)
    @info "Loading CIFAR10 data..."
    train, test = cifar10_data()

    @info "Training..."
    (trained, losses, n) = trainwith(model, train, tparam)
    
    @info "Model:" trained

    @info "Accuracy:" accuracy(trained, train, tparam) accuracy(trained, test, tparam)

    # output a plot of loss
    plot(losses; xaxis=(:log10, "iteration"),
         yaxis="loss", label="per batch")
    # mean loss for epoch
    plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
          label="epoch mean", dpi=200)
end

function simplemodel()
    Chain(MLUtils.flatten,
          Dense(32^2 * 3 => 200, relu),
          Dense(200 => 150, relu),
          Dense(150 => 10),
          softmax)
end

trainandtest(simplemodel(),
             trainparams(batchsize=32,
                         epochs=10,
                         learnrate=5e-4))

function cnnmodel()
    Chain(
        # 1
        Conv((3,3), 3 => 32; pad=SamePad(), stride=1),
        BatchNorm(32, rrelu),
        # 2
        Conv((3,3), 32 => 32; pad=SamePad(), stride=2),
        BatchNorm(32, rrelu),
        # 3
        Conv((3,3), 32 => 64; pad=SamePad(), stride=1),
        BatchNorm(64, rrelu),
        # 4
        Conv((3,3), 64 => 64, pad = SamePad(), stride=2),
        BatchNorm(64, rrelu),
        # 5
        MLUtils.flatten,
        Dense(4096 => 128),
        BatchNorm(128, rrelu),
        Dropout(0.5),
        # 6
        Dense(128 => 10),
        softmax
    )
end

trainandtest(cnnmodel(),
             trainparams(batchsize=32,
                         epochs=10,
                         learnrate=5e-4))
