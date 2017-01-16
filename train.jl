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
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--nogpu"; action=:store_true)
        ("--noisedim"; arg_type=Int64; default=100)
        ("--wdinit"; arg_type=Float32; default=Float32(0.005))
        ("--wginit"; arg_type=Float32; default=Float32(0.05))
        ("--nscale"; arg_type=Float32; default=Float32(0.05))
        ("--dunits"; arg_type=Int64; nargs=2; default=[240,240])
        ("--gunits"; arg_type=Int64; nargs=2; default=[1200,1200])
        ("--maxouts"; arg_type=Int64; nargs=2; default=[5,5])
        ("--epochs"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float32; default=Float32(0.01))
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--h0drop"; arg_type=Float32; default=Float32(0.0))
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
    wd = initD(atype, o[:maxouts], o[:dunits], o[:wdinits], size(dtrn[1][1],1))
    wg = initG(atype, o[:gunits], o[:wginits], o[:noisedim], size(dtrn[1][1],1))

    # gradient check
    if o[:gcheck] > 0
        x = convert(atype, dtrn[1][1])
        z = sample_noise(atype, size(x,2), o[:noisedim], o[:nscale])
        gradcheck(loss1, wd, wg, x, z; gcheck=o[:gcheck])
        gradcheck(loss2, wg, wd, z; gcheck=o[:gcheck])
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
    for epoch = 1:o[:epochs]
        for k = 1:nbatches
            x = convert(atype, dtrn[k][1])
            train!(wd, wz, x, optd, optg, pdrops)
        end
    end
end

# one minibatch training
function train!(wd, wz, x, optd, optg, atype, noisedim, nscale, pdrops)
    z = sample_noise(atype, size(x,2), noisedim, nscale)
    gloss = lossgradient1(wd, wz, x, z)
    map!(k -> update!(wd[k], gloss[k], optd[k]), 1:length(wd))
    z = sample_noise(atype, size(x,2), noisedim, nscale)
    gloss = lossgradient2(wz, wd, z)
    map!(k -> update!(wg[k], gloss[k], optg[k]), 1:length(wg))
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
