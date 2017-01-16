using Knet
using ArgParse
using JLD
using AutoGrad
using GZip

include("data.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "General Adversarial Networks implementation in Knet on MNIST data.",
        " by Ilker Kesen [ikesen16_at_ku.edu.tr], 2017.")

    @add_arg_table s begin
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--nogpu"; action=:store_true)
        ("--noisedim"; arg_type=Int64; default=100)
        ("--wdinit"; arg_type=Float32; default=Float32(0.005))
        ("--wginit"; arg_type=Float32; default=Float32(0.05))
        ("--nscale"; arg_type=Float32; default=Float32(1.25))
        ("--dunits"; arg_type=Int64; nargs=2; default=[240,240])
        ("--gunits"; arg_type=Int64; nargs=2; default=[1200,1200])
        ("--maxouts"; arg_type=Int64; nargs=2; default=[5,5])
        ("--epochs"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float32; default=Float32(0.01))
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--h0drop"; arg_type=Float32; default=Float32(0.2))
        ("--h1drop"; arg_type=Float32; default=Float32(0.0))
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # load data
    (xtrn,xtst,ytrn,ytst)=loaddata()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    dtst = minibatch(xtst, ytst, o[:batchsize])

    # get parameters
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    maxouts, noisedim, nscale = o[:maxouts], o[:noisedim], o[:nscale]
    wd = initD(atype, o[:maxouts], o[:dunits], o[:wdinit], size(dtrn[1][1],1))
    wg = initG(atype, o[:gunits], o[:wginit], o[:noisedim], size(dtrn[1][1],1))
    pdrops = Dict("h0" => o[:h0drop], "h1" => o[:h1drop])

    # gradient check
    if o[:gcheck] > 0
        x = convert(atype, dtrn[1][1])
        z = sample_noise(atype, size(x,2), o[:noisedim], o[:nscale])
        gradcheck(loss, wd, wg, x, z; gcheck=o[:gcheck])
        gradcheck(loss, wg, wd, z; gcheck=o[:gcheck])
    end

    # initialize optimization params, using ADAM
    optd = Array(Any, length(wd))
    optg = Array(Any, length(wg))
    for k = 1:length(wd)
        optd[k] = Adam(wd[k]; lr=o[:lr])
    end
    for k = 1:length(wg)
        optg[k] = Adam(wg[k]; lr=o[:lr])
    end

    # training
    nbatches = length(dtrn)
    prevloss = Inf
    for epoch = 1:o[:epochs]
        tic()
        for k = 1:nbatches
            x = convert(atype, dtrn[k][1])
            train!(wd, wg, x, optd, optg, atype, noisedim, nscale, maxouts, pdrops)
        end
        lossval1 = test(wd, wg, dtrn, atype, noisedim, nscale, maxouts)
        lossval2 = test(wd, wg, dtst, atype, noisedim, nscale, maxouts)
        @printf("\nepoch: %d, loss: %g/%g\n", epoch, lossval1, lossval2)
        flush(STDOUT)
        if o[:savefile] != nothing && lossval2 < prevloss
            save(o[:savefile],
                 "wd", map(Array, wd),
                 "wg", map(Array, wg),
                 "noisedim", noisedim,
                 "nscale", nscale,
                 "maxouts", maxouts)
            @printf("Model saved.\n"); flush(STDOUT)
        end
        toc()
    end
end

# one minibatch training
function train!(wd, wg, x, optd, optg, atype, noisedim, nscale, maxouts, pdrops)
    # discriminator training
    z = sample_noise(atype, size(x,2), noisedim, nscale)
    gloss = lossgradient(wd, wg, x, z; maxouts=maxouts, pdrops=pdrops)
    for k = 1:length(wd)
        update!(wd[k], gloss[k], optd[k])
    end

    # generator training
    z = sample_noise(atype, size(x,2), noisedim, nscale)
    gloss = lossgradient(wg, wd, z; maxouts=maxouts, pdrops=pdrops)
    for k = 1:length(wg)
        update!(wg[k], gloss[k], optg[k])
    end
end

function test(wd, wg, data, atype, noisedim, nscale, maxouts)
    total, count = 0, 0
    nbatches = length(data)
    for k = 1:nbatches
        x = convert(atype, data[k][1])
        z = sample_noise(atype, size(x,2), noisedim, nscale)
        total += loss(wd,wg,x,z)
        count += 1
    end
    return total/count
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
