"""

julia dcgan.jl --outdir ~/dcgan-out
julia dcgan.jl -h # to see all other script options

This example implements a DCGAN (Deep Convolutional Generative Adversarial Network) on MNIST dataset. This implemented model is not identical with the original model. LeNet is used as a base to adapt DCGAN to original MNIST data.1

* Paper url: https://arxiv.org/abs/1511.06434

"""
module DCGAN
using Knet
using Knet: params
using Sloth
using Images
using ArgParse
using JLD2, FileIO
using Printf
using Dates
using Statistics
using Random

include(Knet.dir("data","mnist.jl"))
include(Knet.dir("data","imagenet.jl"))

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    D = Discriminator(o[:atype])
    G = Generator(o[:zdim], o[:atype])
    xtrn,ytrn,xtst,ytst = mnist()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])
    init_optimizers!(D, o[:optim])
    init_optimizers!(G, o[:optim])
    z = sample_noise(o[:atype],o[:zdim],prod(o[:gridsize]))

    if o[:outdir] != nothing && !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"generations"))
    end

    # training
    println("training started..."); flush(stdout)
    for epoch = 1:o[:epochs]
        dlossval = glossval = 0
        @time for (x,y) in dtrn
            input = 2 .* x .- 1
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            dlossval += train_discriminator!(D,G,input,noise)
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            glossval += train_generator!(G,D,noise)
        end
        dlossval /= length(dtrn); glossval /= length(dtrn)
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval))
        flush(stdout)

        # save models and generations
        if o[:outdir] != nothing
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],"generations",filename)
            plot_generations(G; z=z, savefile=filepath,
                             scale=o[:gridscale], gridsize=o[:gridsize])

            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
        end
    end

    return D, G
end

function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Deep Convolutional Generative Adversarial Networks on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=100; help="batch size")
        ("--zdim"; arg_type=Int; default=100; help="noise dimension")
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gridsize"; arg_type=Int; nargs=2; default=[8,8])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;lr=0.0002, beta1=0.5)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(Meta.parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end


function leaky_relu(x, alpha=0.2)
    pos = max.(0,x)
    neg = min.(0,x) .* alpha
    return pos + neg
end


function sample_noise(atype,zdim,nsamples,mu=0.5,sigma=0.5)
    noise = convert(atype, randn(zdim,nsamples))
    normalized = (noise .- mu) ./ sigma
end


mutable struct DLayer1
    conv
    bnorm
end


function (l::DLayer1)(x; stride=1, padding=0, alpha=0.2, training=true)
    y = l.conv(x; stride=stride, padding=padding)
    y = l.bnorm(y; training=training)
    y = pool(y; mode=2)
end


function DLayer1(ci::Int, co::Int, k::Int; atype=Array{Float64})
    conv = Conv(ci, co, k; atype=atype)
    bnorm = BatchNorm(co; atype=atype)
    return DLayer1(conv, bnorm)
end


mutable struct DLayer2
    linear
    bnorm
end


function (l::DLayer2)(x; alpha=0.2, training=true)
    y = l.linear(x)
    y = l.bnorm(y; training=training)
    y = leaky_relu(y, alpha)
end


function DLayer2(input_dim::Int, output_dim::Int; atype=Array{Float64})
    linear = Linear(input_dim, output_dim; atype=atype, bias=false)
    bnorm = BatchNorm(output_dim; atype=atype)
    return DLayer2(linear, bnorm)
end


mutable struct Discriminator
    dlayer1a
    dlayer1b
    dlayer2a
    softlayer
end


function (model::Discriminator)(x; alpha=0.2, training=true)
    y = x
    y = model.dlayer1a(y; training=training)
    y = model.dlayer1b(y; training=training)
    y = reshape(y, 800, size(y, 4))
    y = model.dlayer2a(y; training=training)
    y = sigm.(model.softlayer(y))
end


function Discriminator(atype)
    dlayer1a = DLayer1(1, 20, 5; atype=atype)
    dlayer1b = DLayer1(20, 50, 5; atype=atype)
    dlayer2a = DLayer2(800, 500; atype=atype)
    softlayer = Linear(500, 1; atype=atype)

    return Discriminator(dlayer1a, dlayer1b, dlayer2a, softlayer)
end


function dloss(D::Discriminator, real_images, fake_images)
    yreal = D(real_images)
    real_loss = -mean(log.(yreal))
    yfake = D(fake_images)
    fake_loss = -mean(log.(1 .- yfake))
    return real_loss+fake_loss
end


mutable struct GLayerA
    linear
    bnorm
end


function (l::GLayerA)(x; training=true)
    y = l.linear(x)
    y = l.bnorm(y; training=true)
    y = relu.(y)
end


function GLayerA(input_dim::Int, output_dim::Int; atype=Array{Float64})
    linear = Linear(input_dim, output_dim; atype=atype, bias=false)
    bnorm = BatchNorm(output_dim; atype=atype)
    return GLayerA(linear, bnorm)
end


mutable struct GLayerB
    deconv
    bnorm
end


function (l::GLayerB)(x; stride=2, training=true)
    y = l.deconv(x; stride=stride)
    y = l.bnorm(y; training=training)
end


function GLayerB(ci::Int, co::Int, k::Int; atype=Array{Float64})
    deconv = Deconv(ci, co, k; atype=atype, bias=false)
    bnorm = BatchNorm(co; atype=atype)
    return GLayerB(deconv, bnorm)
end


mutable struct Generator
    glayer1a
    glayer2a
    glayer3b
    glayer4c
    glayer5b
    deconv
end


function (model::Generator)(z; training=true)
    y = model.glayer1a(z; training=training)
    y = model.glayer2a(y; training=true)
    y = reshape(y, 4, 4, 50, size(y, 2))
    y = model.glayer3b(y; training=training)
    y = relu.(model.glayer4c(y; stride=1, training=training))
    y = model.glayer5b(y; training=training)
    y = tanh.(model.deconv(y))
    return y
end


function Generator(zdim, atype)
    glayer1a = GLayerA(zdim, 500; atype=atype)
    glayer2a = GLayerA(500, 800; atype=atype)
    glayer3b = GLayerB(50, 50, 2; atype=atype)
    glayer4c = GLayerB(50, 20, 5; atype=atype)
    glayer5b = GLayerB(20, 20, 2, atype=atype)
    deconv = Deconv(20, 1, 5; atype=atype)

    return Generator(glayer1a, glayer2a, glayer3b, glayer4c, glayer5b, deconv)
end


function gloss(G::Generator, D::Discriminator, noise)
    fake_images = G(noise)
    ypred = D(fake_images)
    return -mean(log.(ypred))
end


function train_discriminator!(
    D::Discriminator, G::Generator, real_images, noise)
    fake_images = G(noise; training=true)
    nsamples = div(length(real_images),784)
    J = @diff dloss(D, real_images, fake_images)
    for p in params(D)
        g = grad(J, p)
        update!(value(p), g, p.opt)
    end
    return value(J)
end


function train_generator!(G::Generator, D::Discriminator, noise)
    J = @diff gloss(G, D, noise)
    for p in params(G)
        g = grad(J, p)
        update!(value(p), g, p.opt)
    end
    return value(J)
end

function plot_generations(
    G::Generator; z=nothing, gridsize=(8,8), scale=1.0, savefile=nothing)
    if z == nothing
        nimg = prod(gridsize)
        zdim = size(G.deconv.glayer1a.w, 2)
        atype = G.deconv.w isa KnetArray ? KnetArray{Float32} : Array{Float32}
        z = sample_noise(atype, zdim, nimg)
    end
    output = Array(0.5 .* (1 .+ G(z; training=false)))
    images = map(i->output[:,:,:,i], 1:size(output,4))
    grid = make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

splitdir(PROGRAM_FILE)[end] == "dcgan.jl" && main(ARGS)

end # module
