import mlx.core as mx

def DepthToSpace(x: mx.array, blocksize, mode="DCR"):
    assert x.ndim == 4, "DepthToSpace only supports 4d input"

    b, c, h, w = x.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = x.reshape(tmpshape)
        transposed = mx.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        # assert mode == "CRD"
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = x.reshape(tmpshape)
        transposed = mx.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    return mx.reshape(transposed, finalshape)

def SpaceToDepth(x: mx.array, blocksize:int):
    assert x.ndim == 4, "SpaceToDepth only supports 4d input"
    b, C, H, W = x.shape
    tmpshape = (
        b,
        C,
        H // blocksize,
        blocksize,
        W // blocksize,
        blocksize,
    )
    reshaped = x.reshape(tmpshape).transpose([0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        C * blocksize * blocksize,
        H // blocksize,
        W // blocksize,
    )
    return reshaped.reshape(finalshape).astype(x.dtype)
    