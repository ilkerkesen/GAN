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

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end
