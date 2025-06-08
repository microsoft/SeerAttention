import torch
import triton
import triton.language as tl

@triton.jit
def _maxpool2d_forward_kernel(
    X, 
    OUT_MAX, 
    MAX_IDX,
    cu_seqlen,
    stride_xm, stride_xh, stride_xd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_iz, stride_ih, stride_im, stride_id,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)
    
    # Each program handles one block in the S dimension
    start_s = pid * BLOCK_SIZE_M

    cu_start = tl.load(cu_seqlen + b).to(tl.int32)
    cu_end = tl.load(cu_seqlen + b + 1).to(tl.int32)
    seq_len = cu_end - cu_start

    if(start_s >= seq_len):
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    s_idx = start_s + offs_m
    mask_s = s_idx < seq_len

    
    x_ptrs = X + (cu_start + s_idx[:, None]) * stride_xm + h * stride_xh  + offs_d[None, :] * stride_xd
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=-float('inf'))
    
    max_x_vals = tl.max(x_vals, axis=0)
    max_x_idx = tl.argmax(x_vals, axis=0)
    
    # Write results
    out_offs =  b * stride_oz + h * stride_oh + pid * stride_om + offs_d * stride_od
    tl.store(OUT_MAX + out_offs, max_x_vals)
    tl.store(MAX_IDX + out_offs, max_x_idx)


@triton.jit
def _maxpool2d_backward_kernel(
    GRAD_OUT,
    IDX,
    GRAD_IN,
    cu_seqlen,
    stride_goz, stride_goh, stride_gom, stride_god,
    stride_gim, stride_gih, stride_gid,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)

    start_s = pid * BLOCK_SIZE_M
    cu_start = tl.load(cu_seqlen + b).to(tl.int32)
    cu_end = tl.load(cu_seqlen + b + 1).to(tl.int32)
    seq_len = cu_end - cu_start

    if(start_s >= seq_len):
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)


    grad_out_ptrs = GRAD_OUT + b * stride_goz + h * stride_goh + pid * stride_gom + offs_d[None, :] * stride_god
    grad_out = tl.load(grad_out_ptrs)

    idx_ptrs = IDX + b * stride_goz + h * stride_goh + pid * stride_gom + offs_d[None, :] * stride_god
    idx = tl.load(idx_ptrs)

    grad_in_ptrs = GRAD_IN + h * stride_gih + (idx + start_s + cu_start) * stride_gim + offs_d[None, :] * stride_gid
    tl.store(grad_in_ptrs, grad_out)


class TritonMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cu_seqlen, max_seqlen, block_size):
        # Output shape after ceil_mode with stride=block_size

        B = cu_seqlen.shape[0] - 1
        nnz, H, D = x.shape[0], x.shape[1], x.shape[2]    
        out_s = (max_seqlen + block_size - 1) // block_size

        x = x.contiguous()


        out = torch.zeros((B, H, out_s, D), device=x.device, dtype=x.dtype)
        idx = torch.zeros_like(out, dtype=torch.int32)
        
        grid = (B, H, out_s)  # One program per block along the S dimension

        with torch.cuda.device(x.device.index): 

            _maxpool2d_forward_kernel[grid](
                x, 
                out, 
                idx,
                cu_seqlen,
                x.stride(0), x.stride(1), x.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
                block_size, 
                D
            )
        
        ctx.save_for_backward(idx, cu_seqlen)
        ctx.shape = (B, H, D, max_seqlen, nnz, block_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        idx, cu_seqlen = ctx.saved_tensors
        B, H, D, max_seqlen, nnz, block_size = ctx.shape
        
        grad_out = grad_out.contiguous()
        idx = idx.contiguous()
        grad_in = torch.zeros((nnz, H, D), device=grad_out.device, dtype=grad_out.dtype)
        
        grid = (B, H, (max_seqlen + block_size - 1) // block_size)

        with torch.cuda.device(grad_out.device.index): 

            _maxpool2d_backward_kernel[grid](
                grad_out, 
                idx, 
                grad_in,
                cu_seqlen,
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                grad_in.stride(0), grad_in.stride(1), grad_in.stride(2),
                block_size,
                D
            )

        return grad_in, None, None, None

def maxpool_varlen(x, cu_seqlen, max_seqlen, block_size):
    return TritonMaxPool2d.apply(x, cu_seqlen, max_seqlen, block_size)




@triton.jit
def _avgpool2d_forward_kernel(
    X, 
    OUT_AVG,
    cu_seqlen,
    stride_xm, stride_xh, stride_xd,
    stride_oz, stride_oh, stride_om, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)


    # Each program handles one block in the S dimension
    start_s = pid * BLOCK_SIZE_M

    cu_start = tl.load(cu_seqlen + b).to(tl.int32)
    cu_end = tl.load(cu_seqlen + b + 1).to(tl.int32)
    seq_len = cu_end - cu_start

    if(start_s >= seq_len):
        return


    if start_s + BLOCK_SIZE_M > seq_len:
        count = seq_len - start_s
    else:
        count = BLOCK_SIZE_M


    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    s_idx = start_s + offs_m
    mask_s = s_idx < seq_len

    x_ptrs = X + (cu_start + s_idx[:, None]) * stride_xm + h * stride_xh  + offs_d[None, :] * stride_xd
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)



    sum_x_vals = tl.sum(x_vals, axis=0)
    avg_x_vals = sum_x_vals / count

    # Write results
    out_offs = b * stride_oz + h * stride_oh + pid * stride_om + offs_d * stride_od
    tl.store(OUT_AVG + out_offs, avg_x_vals)

@triton.jit
def _avgpool2d_backward_kernel(
    GRAD_OUT,
    GRAD_IN,
    cu_seqlen,
    stride_goz, stride_goh, stride_gom, stride_god,
    stride_gim, stride_gih, stride_gid,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)

    start_s = pid * BLOCK_SIZE_M
    cu_start = tl.load(cu_seqlen + b).to(tl.int32)
    cu_end = tl.load(cu_seqlen + b + 1).to(tl.int32)
    seq_len = cu_end - cu_start

    if(start_s >= seq_len):
        return

    if start_s + BLOCK_SIZE_M > seq_len:
        count = seq_len - start_s
    else:
        count = BLOCK_SIZE_M


    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    start_s = start_s + offs_m
    mask_s = start_s < seq_len

    grad_out_ptrs = GRAD_OUT + b * stride_goz + h * stride_goh + pid * stride_gom + offs_d[None, :] * stride_god
    grad_out = tl.load(grad_out_ptrs)




    grad_in_vals = grad_out / count

    grad_in_ptrs = GRAD_IN + h * stride_gih + (start_s[:, None] + cu_start) * stride_gim + offs_d[None, :] * stride_gid
    tl.store(grad_in_ptrs, grad_in_vals, mask=mask_s[:, None])

class TritonAvgPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cu_seqlen, max_seqlen, block_size):
        B = cu_seqlen.shape[0] - 1
        nnz, H, D = x.shape[0], x.shape[1], x.shape[2]    
        out_s = (max_seqlen + block_size - 1) // block_size

        x = x.contiguous()

        out = torch.zeros((B, H, out_s, D), device=x.device, dtype=x.dtype)
        grid = (B, H, out_s)

        with torch.cuda.device(x.device.index): 
            _avgpool2d_forward_kernel[grid](
                x,
                out,
                cu_seqlen,
                x.stride(0), x.stride(1), x.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                block_size,
                D
            )
        
        ctx.save_for_backward(cu_seqlen)
        ctx.shape = (B, H, D, max_seqlen, nnz, block_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        cu_seqlen, = ctx.saved_tensors
        B, H, D, max_seqlen, nnz, block_size = ctx.shape
        grad_in = torch.zeros((nnz, H, D), device=grad_out.device, dtype=grad_out.dtype)
        
        grad_out = grad_out.contiguous()

        grid = (B, H, (max_seqlen + block_size - 1) // block_size)

        with torch.cuda.device(grad_out.device.index): 
            _avgpool2d_backward_kernel[grid](
                grad_out,
                grad_in,
                cu_seqlen,
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                grad_in.stride(0), grad_in.stride(1), grad_in.stride(2),
                block_size,
                D
            )


        return grad_in, None, None, None

def avgpool_varlen(x, cu_seqlen, max_seqlen, block_size):
    return TritonAvgPool2d.apply(x, cu_seqlen, max_seqlen, block_size)