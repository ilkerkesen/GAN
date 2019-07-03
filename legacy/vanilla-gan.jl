using Knet
using ArgParse
using AutoGrad
using GZip
using Images
using ImageMagick

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "General Adversarial Networks implementation in Knet on MNIST data.",
        " by Ilker Kesen [ikesen16_at_ku.edu.tr], 2017.")

    @add_arg_table s begin
        ("--outdir"; default=nothing; help="generations save dir")
        ("--nogpu"; action=:store_true)
        ("--dnet"; nargs='+'; arg_type=Int; default=[128])
        ("--gnet"; nargs='+'; arg_type=Int; default=[128])
        ("--zdim"; arg_type=Int64; default=100)
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--epochs"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float32; default=Float32(0.001))
        ("--adam"; action=:store_true; help="adam optimizer")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--gridsize"; arg_type=Int64; nargs=2; default=[8,8])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--activations"; nargs=3; default=["tanh","tanh","tanh"])
        ("--dropouts"; arg_type=Float64; nargs=3; default=[0.5,0.5,0.5])
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])
    o[:activations] = map(x->eval(parse(x)), o[:activations])
    xscale = o[:activations][end] == tanh ? (255/2) : 255
    xnorm  = o[:activations][end] == tanh ? 1 : 0

    # load data
    (xtrn,xtst,ytrn,ytst)=loaddata()
    trn = minibatch(xtrn, ytrn, o[:batchsize]; xscale=xscale, xnorm=xnorm)
    tst = minibatch(xtst, ytst, o[:batchsize]; xscale=xscale, xnorm=xnorm)

    # get parameters
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    xdim = size(trn[1][1],1)
    wd, wg = initweights(atype, xdim, o[:zdim], o[:dnet], o[:gnet])

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
    if o[:outdir] != nothing
        out = generate(wg,o)
        png = makegrid(out; gridsize=o[:gridsize], scale=o[:gridscale])
        filename = @sprintf("%04d.png",0)
        save(joinpath(o[:outdir],filename), png)
    end

    # training, loss1 => discriminator, loss2 => generator
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        loss1 = loss2 = 0
        for i = 1:length(trn)
            x = convert(atype, trn[i][1])
            z = convert(atype, sample(x))

            losses = train!(wd,wg,x,z,o,optd,optg)
            loss1 += losses[1]; loss2 += losses[2]
        end

        loss1, loss2 = test(wd,wg,trn,o)
        @printf("epoch: %d, losses: %g/%g\n", epoch, loss1, loss2)
        flush(STDOUT)
        if o[:outdir] != nothing
            out = generate(wg,o)
            png = makegrid(out; gridsize=o[:gridsize], scale=o[:gridscale])
            filename = @sprintf("%04d.png",epoch)
            save(joinpath(o[:outdir],filename), png)
        end
    end
end

function dropout(x,d)
    if d > 0
        return x .* (rand!(similar(AutoGrad.getval(x))) .> d) * (1/(1-d))
    else
        return x
    end
end

function D(w,x; f1=tanh, pdrop=0.0)
    for k = 1:2:length(w)-2
        x = f1(w[k] * x .+ w[k+1])
        x = dropout(x, pdrop)
    end
    x = w[end-1] * x .+ w[end]
end

function G(w,z; f2=tanh, f3=tanh, pdrop=0.5)
    x = z
    for k = 1:2:length(w)-2
        x = f2(w[k] * x .+ w[k+1])
        x = dropout(x, pdrop)
    end
    x = f3(w[end-1] * x .+ w[end])
end

function initweights(atype, xdim, zdim, dnet, gnet)
    # discriminator weights
    wd = Array(Any, 2*(1+length(dnet)))
    x = xdim
    for (k,y) in enumerate([dnet..., 2])
        wd[2k-1] = xavier(y,x)
        wd[2k] = zeros(y,1)
        x = y
    end

    # generator weights
    wg = Array(Any, 2*(1+length(gnet)))
    x = zdim
    for (k,y) in enumerate([gnet..., xdim])
        wg[2k-1] = xavier(y,x)
        wg[2k] = zeros(y,1)
        x = y
    end

    map(x->convert(atype, x), wd), map(x->convert(atype, x), wg)
end

# loss for discriminator network
function dloss(wd,wg,x,z,labels,fs,pdrops,values=[])
    real = x
    fake = G(wg,z; pdrop=pdrops[2], f2=fs[2], f3=fs[3])
    data = hcat(real,fake)
    pred = logp(D(wd,data; pdrop=pdrops[1], f1=fs[1]), 1)
    val  = -sum(pred .* labels) / size(data,2)
    push!(values, val)
    return val
end

# loss for generator network
function gloss(wg,wd,z,labels,fs,pdrops,values=[])
    data = G(wg,z; pdrop=pdrops[2], f2=fs[2], f3=fs[3])
    pred = logp(D(wd,data; pdrop=pdrops[1], f1=fs[1]), 1)
    val  = -sum(pred .* labels) / size(data,2)
    push!(values, val)
    return val
end

dlossgradient = grad(dloss)
glossgradient = grad(gloss)

function sample_noise(atype, batchsize, dimension, scale=1)
    return convert(atype, scale*randn(dimension,batchsize))
end

initopt(w,lr) = map(x->Adam(;lr=lr), w)

function train!(wd,wg,x,z,o,optd,optg)
    val = []
    atype = typeof(AutoGrad.getval(wd[1]))
    yx = zeros(2,size(x,2)); yx[1,:] = 1; yx = convert(atype, yx)
    yz = zeros(2,size(z,2)); yz[2,:] = 1; yz = convert(atype, yz)
    g = dlossgradient(wd,wg,x,z,hcat(yx,yz),o[:activations],o[:dropouts],val)
    for k = 1:length(wd); update!(wd[k], g[k], optd[k]); end
    g = glossgradient(wg,wd,z,yx,o[:activations],o[:dropouts],val)
    for k = 1:length(wg); update!(wg[k], g[k], optg[k]); end
    return val
end

function test(wd,wg,data,o)
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    sample(x) = sample_noise(atype, size(x,2), o[:zdim])
    loss1 = loss2 = 0
    for (x,y) in data
        x = convert(atype, x)
        z = convert(atype, sample(x))
        yx = zeros(2,size(x,2)); yx[1,:] = 1; yx = convert(atype, yx)
        yz = zeros(2,size(z,2)); yz[2,:] = 1; yz = convert(atype, yz)
        loss1 += dloss(wd,wg,x,z,hcat(yx,yz),o[:activations],o[:dropouts])
        loss2 += gloss(wg,wd,z,yx,o[:activations],o[:dropouts])
    end
    return (loss1/length(data),loss2/length(data))
end

function generate(wg,o)
    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    sample(n) = sample_noise(atype, n, o[:zdim])
    ninstances = o[:gridsize][1] * o[:gridsize][2]
    z = convert(atype, sample(ninstances))
    y = G(wg,z; pdrop=o[:dropouts][end])
    y = convert(Array{Float64}, y)
    y = (y+1)/2
    return min(1,max(0,y))
end

function makegrid(y; gridsize=[4,4], scale=2.0, shape=(28,28))
    y = reshape(y, shape..., size(y,2))
    y = map(x->y[:,:,x]', [1:size(y,3)...])
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
    out = zeros(outdims...)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] = 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] = 1.0; end

    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 2
        else
            y0 = y1+2
        end
    end

    return convert(Array{Float64,2}, map(x->isnan(x)?0:x, out))
end

function minibatch(
    x, y, batchsize;
    atype=Array{Float32}, xrows=784, yrows=10, xscale=255/2, xnorm=1)
    xbatch(a)=convert(atype, reshape(a./xscale-xnorm, xrows, div(length(a),xrows)))
    ybatch(a)=(a[a.==0]=10; convert(atype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
    return data
end

function loaddata()
    info("Loading MNIST...")
    gzload("train-images-idx3-ubyte.gz")[17:end],
    gzload("t10k-images-idx3-ubyte.gz")[17:end],
    gzload("train-labels-idx1-ubyte.gz")[9:end],
    gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = read(f)
    close(f)
    return(a)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
