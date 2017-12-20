for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))

module GAN2D
using Knet
using Images
using ArgParse

include(Pkg.dir("Knet","data","mnist.jl"))

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load model & data
    wd, wg = initweights(o[:atype])
    xtrn,ytrn,xtst,ytst = Main.mnist()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)

    # training

end

function parse_options(args)
    s = ArgParseSettings()
    s.description = "Conditional Generative Adversarial Networks on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=100; help="batch size")
        ("--zdim"; arg_type=Int; default=100; help="noise dimension")
        ("--epochs"; arg_type=Int; default=3; help="number of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    return o
end

function leaky_relu(x, alpha=0.01)
    x1 = max.(0,x)
    x2 = min.(0,x) * alpha
    return x1 .+ x2
end

function initweights(atype)
    wd, wg = initwd(atype), initwg(atype)
end

function initwd(atype)
    N = 4
    w = Array{Any}(N)
    w[1] = convert(atype, xavier(5,5,1,64))
    w[2] = convert(atype, zeros(1,1,64,1))
    w[3] = convert(atype, xavier(5,5,64,128))
    w[4] = convert(atype, zeros(1,1,128,1))
    w[5] = convert(atype, xavier(1024,2048))
    w[6] = convert(atype, zeros(1024,1))
    w[7] = convert(atype, xavier(2,1024))
    w[8] = convert(atype, zeros(2,1))
    return w
end

function initwg(atype=Array{Float32}, zdim=100, winit=0.01)
    N = 6
    w = Array{Any}(N)
    w[1] = convert(atype, xavier(6*6*128, zdim))
    w[2] = convert(atype, zeros(1,1,128,1))
    w[3] = convert(atype, xavier(4,4,64,128))
    w[4] = convert(atype, zeros(1,1,64,1))
    w[5] = convert(atype, xavier(2,2,1,64))
    w[6] = convert(atype, zeros(1,1,1,1))
    return w
end

function gnet(wg,z)
    x = tanh.(wg[1] * z .+ wg[2])
    x = reshape(x, 6,6,128,size(z,2))
    x = deconv4(wg[3],x; stride=2) .+ wg[4]
    x = deconv4(wg[5],x; stride=2) .+ wg[6]
    x = tanh.(x)
end

function dnet(w,x0,ygold)
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

splitdir(PROGRAM_FILE)[end] == "cgan2d.jl" && main(ARGS)

end # module
