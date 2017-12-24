for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))

module GAN2D
using Knet
using Images
using ArgParse
using JLD2, FileIO

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    wd, wg = load_weights(o[:atype], o[:zdim], o[:loadfile])
    xtrn,ytrn,xtst,ytst = Main.mnist()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])
    optd = map(wi->Sgd(;lr=0.005), wd)
    optg = map(wi->Adam(), wg)

    # TODO: gradcheck
    # gradcheck(...)

    # training
    println("training started..."); flush(STDOUT)
    o[:outdir] != nothing && !isdir(o[:outdir]) && mkpath(o[:outdir])
    for epoch = 1:o[:epochs]
        dlossval = glossval = 0
        for (x,y) in dtrn
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            dlossval += train_discriminator!(wd,wg,2x-1,y,noise,optd,o)
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            glossval += train_generator!(wg,wd,noise,y,optg,o)
        end
        dlossval /= length(dtrn); glossval /= length(dtrn)
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval))
        flush(STDOUT)

        # save generation
        if o[:outdir] != nothing
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],filename)
            plot_generations(wg,filepath)
        end
    end

    o[:savefile] != nothing && save_weights(o[:savefile], wd, wg)
    return wd,wg
end

function parse_options(args)
    s = ArgParseSettings()
    s.description = "Deconvolutional Generative Adversarial Networks on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=50; help="batch size")
        ("--zdim"; arg_type=Int; default=200; help="noise dimension")
        ("--pdrop"; arg_type=Float64; default=.5; help="dropout probability")
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--optim"; default="Adam(;gclip=5.0)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--savefile"; default=nothing; help="file to save trained models")
        ("--outdir"; default=nothing; help="output directory for generations")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    o[:outdir] = abspath(o[:outdir])
    return o
end

function leaky_relu(x, alpha=0.01)
    x1 = max.(0,x)
    x2 = min.(0,x) * alpha
    return x1 .+ x2
end

function load_weights(atype,zdim,loadfile=nothing)
    if loadfile == nothing
        wd, wg = initwd(atype), initwg(atype,zdim)
    else
        @load loadfile wd wg
        wd = convert_weights(wd, atype)
        wg = convert_weights(wg, atype)
    end
    return wd, wg
end

function initwd(atype)
    N = 8
    w = Array{Any}(N)
    w[1] = xavier(5,5,1,10)
    w[2] = zeros(1,1,10,1)
    w[3] = xavier(5,5,10,20)
    w[4] = zeros(1,1,20,1)
    w[5] = xavier(1024,320)
    w[6] = zeros(1024,1)
    # w[7] = xavier(11,1024)
    # w[8] = zeros(11,1)
    w[7] = xavier(2,1024)
    w[8] = zeros(2,1)
    return convert_weights(w,atype)
end

function initwg(atype=Array{Float32}, zdim=100, winit=0.01)
    N = 6
    w = Array{Any}(N)
    w[1] = xavier(6*6*20, zdim)
    w[2] = zeros(6*6*20,1)
    w[3] = xavier(4,4,10,20)
    w[4] = zeros(1,1,10,1)
    w[5] = xavier(2,2,1,10)
    w[6] = zeros(1,1,1,1)
    return convert_weights(w,atype)
end

function sample_noise(atype,zdim,nsamples,a=2,b=1)
    return convert(atype, a*rand(zdim, nsamples)-b)
end

function plot_generations(wg, savefile=nothing; gridsize=(5,5), scale=2.0)
    zdim = size(wg[1],2)
    atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
    m, n = gridsize
    nimg = m*n
    z = sample_noise(atype,zdim,nimg)
    generated = Array(0.5*(1+gnet(wg, z)))
    grid = Main.mnistgrid(generated; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

function gnet(wg,z; pdrop=0.5, training=true)
    x = tanh.(wg[1] * z .+ wg[2])
    x = reshape(x, 6,6,20,size(z,2))
    x = deconv4(wg[3],x; stride=2) .+ wg[4]
    x = deconv4(wg[5],x; stride=2) .+ wg[6]
    x = tanh.(x)
end

function dnet(w,x0; pdrop=0.5)
    x = conv4(w[1], x0) .+ w[2]
    x = tanh.(x)
    x = pool(x; mode=2)
    x = conv4(w[3], x) .+ w[4]
    x = tanh.(x)
    x = pool(x; mode=2)
    x = mat(x)
    x = w[5] * x .+ w[6]
    x = tanh.(x)
    x = w[7] * x .+ w[8]
end

function dloss(w,x0,ygold)
    ypred = dnet(w,x0)
    return nll(ypred, ygold)
end

dlossgradient = gradloss(dloss)

function train_discriminator!(wd,wg,real_images,real_labels,noise,optd,o)
    fake_images = gnet(wg,noise; training=true)
    nsamples = div(length(real_images),784)
    input = begin
        x1 = reshape(real_images, 784, size(real_images,4))
        x2 = reshape(fake_images, 784, size(fake_images,4))
        concat = hcat(x1,x2)
        resized = reshape(concat, 28, 28, 1, size(concat,2))
    end
    labels = hcat(1*ones(Int64, 1, nsamples),
                  # reshape(real_labels, 1, nsamples),
                  2*ones(Int64, 1, nsamples))
    gradients, lossval = dlossgradient(wd,input,labels)
    update!(wd, gradients, optd)
    return lossval
end

function gloss(wg,wd,noise,ygold)
    fake_images = gnet(wg,noise)
    ypred = dnet(wd,fake_images)
    return nll(ypred, ygold)
end

glossgradient = gradloss(gloss)

function train_generator!(wg,wd,noise,labels,optg,o)
    ygold = ones(Int64, 1, length(labels))
    gradients, lossval = glossgradient(wg,wd,noise,ygold)
    update!(wg,gradients,optg)
    return lossval
end

function convert_weights(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end

function save_weights(savefile,wd,wg)
    save(savefile, "wd", convert_weights(wd), "wg", convert_weights(wg))
end

splitdir(PROGRAM_FILE)[end] == "gan2d.jl" && main(ARGS)

end # module
