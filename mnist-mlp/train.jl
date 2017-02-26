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
        ("--hdim"; arg_type=Int64; default=128)
        ("--zdim"; arg_type=Int64; default=100)
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--epochs"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float32; default=Float32(0.001))
        ("--adam"; action=:store_true; help="adam optimizer")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # load data
    (xtrn,xtst,ytrn,ytst)=loaddata()
    trn = minibatch(xtrn, ytrn, o[:batchsize])
    tst = minibatch(xtst, ytst, o[:batchsize])

    # get parameters
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    wd, wg = initweights(atype, size(trn[1][1],1), o[:hdim], o[:zdim])

    # gradient check
    if o[:gcheck] > 0
        x = convert(atype, trn[1][1])
        z = sample_noise(atype, size(trn[1][1],2), o[:zdim])
        gradcheck(loss, wd, wg, x, z; gcheck=o[:gcheck])
        gradcheck(loss, wg, wd, z; gcheck=o[:gcheck])
    end

    # initialize optimization params, using ADAM
    optd = initopt(wd,o[:lr])
    optg = initopt(wg,o[:lr])
    sample(x) = sample_noise(atype, size(x,2), o[:zdim])

    # performance with random initialization
    loss1, loss2 = test(wd,wg,trn,o)
    @printf("\nepoch: %d, losses: %g/%g [trn]\n", 0, loss1, loss2)
    loss1, loss2 = test(wd,wg,tst,o)
    @printf("epoch: %d, losses: %g/%g [tst]\n", 0, loss1, loss2)
    println(); flush(STDOUT)


    # training, loss1 => discriminator, loss2 => generator
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        loss1 = loss2 = 0
        for i = 1:length(trn)
            x = convert(atype, trn[i][1])
            z = convert(atype, sample(x))

            losses = train!(wd,wg,x,z,optd,optg)
            loss1 += losses[1]; loss2 += losses[2]
        end

        loss1, loss2 = test(wd,wg,trn,o)
        @printf("epoch: %d, losses: %g/%g [trn]\n", epoch, loss1, loss2)
        loss1, loss2 = test(wd,wg,tst,o)
        @printf("epoch: %d, losses: %g/%g [tst]\n", epoch, loss1, loss2)
        println(); flush(STDOUT)
    end

    if o[:savefile] != nothing
        save(o[:savefile],
             "wd", map(Array, wd),
             "wg", map(Array, wg))
        @printf("Model saved.\n"); flush(STDOUT)
    end
end

initopt(w,lr) = map(x->Adam(x;lr=lr), w)

function train!(wd,wg,x,z,optd,optg)
    values = []
    g = dlossgradient(wd,wg,x,z,values)
    for k = 1:length(wd); update!(wd[k], g[k], optd[k]); end
    g = glossgradient(wg,wd,z,values)
    for k = 1:length(wg); update!(wg[k], g[k], optg[k]); end
    return values
end

function test(wd,wg,data,o)
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    sample(x) = sample_noise(atype, size(x,2), o[:zdim])
    loss1 = loss2 = 0
    for (x,y) in data
        x = convert(atype, x)
        z = convert(atype, sample(x))
        loss1 += dloss(wd,wg,x,z)
        loss2 += gloss(wg,wd,z)
    end
    return (loss1/length(data),loss2/length(data))
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
