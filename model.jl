function dropout(x,d)
    if d > 0
        return x .* (rand!(similar(AutoGrad.getval(x))) .> d) * (1/(1-d))
    else
        return x
    end
end

function D(w,x; maxouts=[5,5])
    h0 = max(map(w1i -> w1i*x, w[1:maxouts[1]])...)
    h1 = max(map(w2i -> w2i*h0, w[maxouts[1]+1:sum(maxouts[2])])...)
    y  = sigm(w[sum(maxouts)+1]*h1)
end

function G(w,z; pdrops=Dict())
    d0 = get(pdrops, "h0", 0.0)
    d1 = get(pdrops, "h1", 0.0)

    h0 = dropout(relu(w[1] * z), d0)
    h1 = dropout(relu(w[2] * h0), d1)
    y  = sigm(w[3] * h1)
end

loss1(wd,wz,x,z) = -(log(D(wd,x)) + log(1-D(wd,G(wz,z)))) / size(z,2)
loss2(wz,z) = log(1-D(G(z))) / size(z,2)

lossgradient1 = grad(loss1)
lossgradient2 = grad(loss2)

function initD(atype, maxouts, units, winit, inputdim)
    w1 = map(i->convert(atype,winit*randn(units[1],inputdim)),1:maxouts[1])
    w2 = map(i->convert(atype,winit*randn(units[2],units[1])),1:maxouts[2])
    w3 = convert(atype, winit*randn(1,units[2]))
    ws = [w1..., w2..., w3]
end

function initG(atype, units, winit, inputdim, outputdim)
    w1 = convert(atype,winit*randn(units[1],inputdim))
    w2 = convert(atype,winit*randn(units[2],units[1]))
    w3 = convert(atype,winit*randn(outputdim,units[2]))
    ws = [w1, w2, w3]
end

function sample_noise(atype, batchsize, dimension, scale)
    return convert(atype, scale*randn(dimension,batchsize))
end
