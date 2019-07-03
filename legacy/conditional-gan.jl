for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

module CGAN
using Knet
using Images
using ArgParse
using JLD2, FileIO

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    wd, wg, md, mg = load_weights(o[:atype], o[:zdim], o[:loadfile])
    xtrn,ytrn,xtst,ytst = Main.mnist()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])
    optd = map(wi->eval(parse(o[:optim])), wd)
    optg = map(wi->eval(parse(o[:optim])), wg)
    z = sample_noise(o[:atype],o[:zdim],prod(o[:gridsize]))
    l0 = map(i->reshape(collect(1:9), 1, 9), 1:9)
    l1 = vec(vcat(l0...))

    if o[:outdir] != nothing && !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"generations"))
    end

    # training
    println("training started..."); flush(STDOUT)
    for epoch = 1:o[:epochs]
        dlossval = glossval = 0
        @time for (x,y) in dtrn
            x0 = 2x-1; x0 = reshape(x0, 784, div(length(x0),784))
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            dlossval += train_discriminator!(wd,wg,md,mg,x0,y,noise,optd,o)
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            glossval += train_generator!(wg,wd,mg,md,noise,y,optg,o)
        end
        dlossval /= length(dtrn); glossval /= length(dtrn)
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval))
        flush(STDOUT)

        # save models and generations
        if o[:outdir] != nothing
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],"generations",filename)
            plot_generations(
                wg, mg, l1; z=z, savefile=filepath,
                scale=o[:gridscale], gridsize=o[:gridsize])

            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            save_weights(filepath,wd,wg,md,mg)
        end
    end

    return wd,wg,md,mg
end

function parse_options(args)
    s = ArgParseSettings()
    s.description = "Conditional GANs on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=100; help="batch size")
        ("--zdim"; arg_type=Int; default=100; help="noise dimension")
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gridsize"; arg_type=Int; nargs=2; default=[9,9])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;lr=0.0002, beta1=0.5)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end

function load_weights(atype,zdim,loadfile=nothing)
    if loadfile == nothing
        wd, md = initwd(atype)
        wg, mg = initwg(atype)
    else
        @load loadfile wd wg md mg
        wd = convert_weights(wd, atype)
        wg = convert_weights(wg, atype)
        md = convert_moments(md, atype)
        mg = convert_moments(mg, atype)
    end
    return wd, wg, md, mg
end

function save_weights(savefile,wd,wg,md,mg)
    save(savefile,
         "wd", convert_weights(wd),
         "wg", convert_weights(wg),
         "md", convert_moments(md),
         "mg", convert_moments(mg))
end

function convert_weights(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end


function convert_moments(moments,atype=Array{Float32})
    clone = map(mi->bnmoments(), moments)
    for k = 1:length(clone)
        if moments[k].mean != nothing
            clone[k].mean = convert(atype, moments[k].mean)
        end

        if moments[k].var != nothing
            clone[k].var = convert(atype, moments[k].var)
        end
    end
    return convert(Array{Any,1}, clone)
end


function leaky_relu(x, alpha=0.2)
    pos = max(0,x)
    neg = min(0,x) * alpha
    return pos + neg
end

function sample_noise(atype,zdim,nsamples,mu=0.5,sigma=0.5)
    noise = convert(atype, randn(zdim,nsamples))
    normalized = (noise-mu)/sigma
end

function initwd(atype, winit=0.01)
    w = Any[]
    m = Any[]

    # fc layers for inputs (x,y)
    push!(w, winit*randn(1024,784))
    push!(w, zeros(1024,1))
    push!(w, winit*randn(1024,10))
    push!(w, zeros(1024,1))

    # fc layers for concatenated
    push!(w, winit*randn(512,2048))
    push!(w, bnparams(512))
    push!(m, bnmoments())
    push!(w, winit*randn(256,512))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    push!(w, winit*randn(2,256))
    push!(w, zeros(2,1))
    return convert_weights(w,atype), m
end

function dnet(w,x,y,m; training=true, pdrop=0.5, alpha=0.2)
    x1 = leaky_relu.(w[1] * x .+ w[2], alpha)
    x2 = leaky_relu.(w[3][:,y] .+ w[4])
    x3 = vcat(x1,x2)
    x4 = leaky_relu.(batchnorm(w[5]*x3, m[1], w[6]), alpha)
    x5 = leaky_relu.(batchnorm(w[7]*x4, m[2], w[8]), alpha)
    x6 = w[9] * x5 .+ w[10]
end

function dloss(w,m,real_images,real_labels,fake_images,fake_labels,ygold)
    yreal = dnet(w,real_images,ygold,m)
    real_loss = nll(yreal, real_labels)
    yfake = dnet(w,fake_images,ygold,m)
    fake_loss = nll(yfake, fake_labels)
    return real_loss + fake_loss
end

dlossgradient = gradloss(dloss)

function train_discriminator!(wd,wg,md,mg,real_images,ygold,noise,optd,o)
    fake_images = gnet(wg,noise,ygold,mg; training=true)
    nsamples = div(length(real_images),784)
    real_labels = ones(Int64, 1, nsamples)
    fake_labels = 2ones(Int64, 1, nsamples)
    gradients, lossval = dlossgradient(
        wd,md,real_images,real_labels,fake_images,fake_labels,ygold)
    update!(wd, gradients, optd)
    return lossval
end

function initwg(atype, winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(256,100))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    push!(w, winit*randn(256,10))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    push!(w, winit*randn(512,512))
    push!(w, bnparams(512))
    push!(m, bnmoments())

    push!(w, winit*randn(1024,512))
    push!(w, bnparams(1024))
    push!(m, bnmoments())

    push!(w, winit*randn(784,1024))
    push!(w, zeros(784,1))

    return convert_weights(w,atype), m
end

function gnet(w,z,y,m; training=true, pdrop=0.5)
    x0 = relu.(batchnorm(w[1]*z, m[1], w[2]))
    x1 = relu.(batchnorm(w[3][:,y], m[2], w[4]))
    x2 = vcat(x0,x1)
    x3 = relu.(batchnorm(w[5]*x2, m[3], w[6]))
    x4 = relu.(batchnorm(w[7]*x3, m[4], w[8]))
    x5 = tanh.(w[9] * x4 .+ w[10])
end

function gloss(wg,wd,mg,md,noise,ygold,labels)
    fake_images = gnet(wg,noise,labels,mg)
    ypred = dnet(wd,fake_images,labels,md)
    return nll(ypred, ygold)
end

glossgradient = gradloss(gloss)

function train_generator!(wg,wd,mg,md,noise,labels,optg,o)
    ygold = ones(Int64, 1, length(labels))
    gradients, lossval = glossgradient(wg,wd,mg,md,noise,ygold,labels)
    update!(wg,gradients,optg)
    return lossval
end

function plot_generations(
    wg, mg, labels; z=nothing, gridsize=(8,8), scale=1.0, savefile=nothing)
    if z == nothing
        nimg = prod(gridsize)
        zdim = size(wg[1],2)
        atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
        z = sample_noise(atype,zdim,nimg)
    end
    output = Array(0.5*(1+gnet(wg,z,labels,mg; training=false)))
    images = map(i->reshape(output[:,i], 28, 28, 1), 1:size(output,2))
    grid = Main.make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

splitdir(PROGRAM_FILE)[end] == "conditional-gan.jl" && main(ARGS)

end # module
