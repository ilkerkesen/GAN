using Knet
using ArgParse
using AutoGrad
using GZip
using Images
using ImageMagick

include("data.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "General Adversarial Networks implementation in Knet on MNIST data.",
        " by Ilker Kesen [ikesen16_at_ku.edu.tr], 2017.")

    @add_arg_table s begin
        ("--outdir"; default=nothing; help="generations save dir")
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

    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
        !isdir(o[:outdir]) && mkdir(o[:outdir])
    end

    # initialize optimization params, using ADAM
    optd = initopt(wd,o[:lr])
    optg = initopt(wg,o[:lr])
    sample(x) = sample_noise(atype, size(x,2), o[:zdim])

    # performance with random initialization
    loss1, loss2 = test(wd,wg,trn,o)
    @printf("\nepoch: %d, losses: %g/%g\n", 0, loss1, loss2)

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
        @printf("epoch: %d, losses: %g/%g\n", epoch, loss1, loss2)
        println(); flush(STDOUT)
        if o[:outdir] != nothing
            out = generate(wg,o)
            png = makegrid(out)
            filename = @sprintf("%04d.png",epoch)
            save(joinpath(o[:outdir],filename), png)
        end

    end
end

function D(w,x)
    h = relu(w[1] * x .+ w[2])
    y = sigm(w[3] * h .+ w[4])
end

function G(w,z)
    h = relu(w[1] * z .+ w[2])
    x = sigm(w[3] * h .+ w[4])
end

function initweights(atype, xdim, hdim, zdim)
    wd = Array(Any, 4)
    wg = Array(Any, 4)

    wd[1] = xavier(hdim,xdim)
    wd[2] = zeros(hdim,1)
    wd[3] = xavier(2,hdim)
    wd[4] = zeros(2,1)

    wg[1] = xavier(hdim,zdim)
    wg[2] = zeros(hdim,1)
    wg[3] = xavier(xdim,hdim)
    wg[4] = zeros(xdim,1)

    return map(x->convert(atype, x), wd), map(x->convert(atype, x), wg)
end

# loss for discriminator network
function dloss(wd,wg,x,z,values=[])
    x1  = sum(log(D(wd,x)))/size(x,2)
    z1  = sum(log(1-D(wd,G(wg,z))))/size(z,2)
    val = -0.5 * (x1+z1)
    push!(values, val)
    return val
end

# loss for generator network
function gloss(wg,wd,z,values=[])
    val = -0.5*sum(log(D(wd,G(wg,z)))) / size(z,2)
    push!(values, val)
    return val
end

dlossgradient = grad(dloss)
glossgradient = grad(gloss)

function sample_noise(atype, batchsize, dimension, scale=1)
    return convert(atype, scale*randn(dimension,batchsize))
end

initopt(w,lr) = map(x->Adam(;lr=lr), w)

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

function generate(wg,o,ninstances=16)
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    sample(n) = sample_noise(atype, n, o[:zdim])
    z = convert(atype, sample(ninstances))
    y = G(wg,z)
end

function makegrid(y; gridsize=(4,4),scale=2, shape=(28,28))
    y = convert(Array{Float64}, y)
    y = reshape(y, shape..., size(y,2))
    y = map(x->y[:,:,x]', [1:size(y,3)...])
    shp = (shape[1]*scale, shape[2]*scale)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx-1,gridy*shp[2]+gridy-1)
    out = zeros(outdims...)
    for k = 1:gridx-1; out[k*(shp[1]+1),:] = 1.0; end
    for k = 1:gridy-1; out[:,k*(shp[2]+1)] = 1.0; end

    xcor = ycor = 1
    x0 = y0 = 1
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        # x0 = x1+2; y0 = y1+2;
        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 1
        else
            y0 = y1+2
        end
    end

    return out
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
