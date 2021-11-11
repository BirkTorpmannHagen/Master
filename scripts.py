from math import floor
def compute_shape_conv2d(indim, kernel_size, stride, padding, dilation=1):
    return floor((indim+2*padding-dilation*(kernel_size-1)-1)/stride+1)

def compute_shape_up(indim, kernel_size, stride, padding, dilation=1, output_padding=0):
    return (indim-1)*stride - 2*padding + dilation*(kernel_size-1)+output_padding+1
if __name__ == '__main__':
    params_down = [
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
    ]

    params_up = [
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
        [4, 2, 1],
    ]
    last = [3,1,9]

    in_dim = 400
    out = None
    for i in params_down:
        out = compute_shape_conv2d(in_dim, *i)
        print(f"{in_dim}>{out}")
        in_dim = out
    for i in params_up:
        out = compute_shape_up(in_dim, *i)
        print(f"{out}<{in_dim}")
        in_dim = out
    out = compute_shape_conv2d(in_dim, *last)
    print(f"\n{in_dim}>{out}")

