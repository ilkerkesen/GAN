function dropout(x,d)
    if d > 0
        return x .* (rand!(similar(AutoGrad.getval(x))) .> d) * (1/(1-d))
    else
        return x
    end
end

function D(wd,x; maxouts=[5,5])
    h0 = max(map(k -> wd[2*k-1]*x  .+ wd[2*k], 1:maxouts[1])...)
    h1 = max(map(k -> wd[2*k-1]*h0 .+ wd[2*k], maxouts[1]+1:sum(maxouts))...)
    k  = 2*sum(maxouts)+1
    y  = sigm(wd[k] * h1 .+ wd[k+1])
end

function G(wg,z; pdrops=Dict())
    d0 = get(pdrops, "h0", 0)
    d1 = get(pdrops, "h1", 0)

    h0 = dropout(relu(wg[1] * z .+ wg[2]), d0)
    h1 = dropout(relu(wg[3] * h0 .+ wg[4]), d1)
    y  = sigm(wg[5] * h1 .+ wg[6])
end

function loss(wd,wg,x,z; maxouts=[5,5], pdrops=Dict())
    -sum(log(D(wd,x; maxouts=maxouts)) +
         log(1-D(wd,G(wg,z; pdrops=pdrops); maxouts=maxouts))) / size(z,2)
end
function loss(wg,wd,z; maxouts=[5,5], pdrops=Dict())
    sum(log(1-D(wd,G(wg,z; pdrops=pdrops); maxouts=maxouts))) / size(z,2)
end
lossgradient = grad(loss)

function initD(atype, maxouts, units, winit, inputdim)
    w = Array(Any, 2*(sum(maxouts)+1))
    outputdim = 1
    for k = 1:maxouts[1]
        w[2*k-1] = convert(atype,winit*randn(units[1],inputdim))
        w[2*k]   = convert(atype,winit*randn(units[1],1))
    end
    for k = 1:maxouts[2]
        w[2*maxouts[1]+2*k-1] = convert(atype,winit*randn(units[2],units[1]))
        w[2*maxouts[1]+2*k]   = convert(atype,winit*randn(units[2],1))
    end
    w[end-1] = convert(atype,winit*randn(outputdim,units[2]))
    w[end]   = convert(atype,winit*randn(outputdim,1))
    return w
end

function initG(atype, units, winit, inputdim, outputdim)
    w = Array(Any, 6)
    w[1] = convert(atype,winit*randn(units[1],inputdim))
    w[2] = convert(atype,winit*randn(units[1],1))
    w[3] = convert(atype,winit*randn(units[2],units[1]))
    w[4] = convert(atype,winit*randn(units[2],1))
    w[5] = convert(atype,winit*randn(outputdim,units[2]))
    w[6] = convert(atype,winit*randn(outputdim,1))
    return w
end

function sample_noise(atype, batchsize, dimension, scale)
    return convert(atype, sort(scale*randn(dimension,batchsize),1))
end
