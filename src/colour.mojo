"""Colour conversion, difference, and appearance kernels over float64 buffers."""

from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import atan2, cbrt, cos, exp, pow, sin, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.141592653589793238462643383279502884
comptime PARALLEL_THRESHOLD = 65536
comptime PARALLEL_CHUNK = 32768


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def sign(v: Float64) -> Float64:
    if v < 0.0:
        return -1.0
    if v > 0.0:
        return 1.0
    return 0.0


def spow(v: Float64, exponent: Float64) -> Float64:
    return sign(v) * pow(abs(v), exponent)


def mat3(
    m: Ptr, x: Float64, y: Float64, z: Float64
) -> Tuple[Float64, Float64, Float64]:
    return (
        m[0] * x + m[1] * y + m[2] * z,
        m[3] * x + m[4] * y + m[5] * z,
        m[6] * x + m[7] * y + m[8] * z,
    )


def lab_f(t: Float64) -> Float64:
    if t > 216.0 / 24389.0:
        return cbrt(t)
    return (841.0 / 108.0) * t + 4.0 / 29.0


def lab_f_simd[W: Int](
    t: SIMD[DType.float64, W]
) -> SIMD[DType.float64, W]:
    return t.gt(216.0 / 24389.0).select(
        cbrt(t), (841.0 / 108.0) * t + 4.0 / 29.0
    )


def lab_f_inverse(t: Float64) -> Float64:
    if t > 6.0 / 29.0:
        return t * t * t
    return (108.0 / 841.0) * (t - 4.0 / 29.0)


def srgb_decode(v: Float64) -> Float64:
    if v <= 0.04045:
        return v / 12.92
    return pow((v + 0.055) / 1.055, 2.4)


def srgb_encode(v: Float64) -> Float64:
    if v <= 0.0031308:
        return 12.92 * v
    return 1.055 * spow(v, 1.0 / 2.4) - 0.055


@export("mcs_srgb_to_xyz")
def mcs_srgb_to_xyz(src_addr: Int, dst_addr: Int, n: Int, decode: Int) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var r = src[3 * i]
        var g = src[3 * i + 1]
        var b = src[3 * i + 2]
        if decode != 0:
            r = srgb_decode(r)
            g = srgb_decode(g)
            b = srgb_decode(b)
        dst[3 * i] = 0.4124 * r + 0.3576 * g + 0.1805 * b
        dst[3 * i + 1] = 0.2126 * r + 0.7152 * g + 0.0722 * b
        dst[3 * i + 2] = 0.0193 * r + 0.1192 * g + 0.9505 * b


@export("mcs_xyz_to_srgb")
def mcs_xyz_to_srgb(src_addr: Int, dst_addr: Int, n: Int, encode: Int) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var x = src[3 * i]
        var y = src[3 * i + 1]
        var z = src[3 * i + 2]
        var r = 3.2406 * x - 1.5372 * y - 0.4986 * z
        var g = -0.9689 * x + 1.8758 * y + 0.0415 * z
        var b = 0.0557 * x - 0.2040 * y + 1.0570 * z
        if encode != 0:
            r = srgb_encode(r)
            g = srgb_encode(g)
            b = srgb_encode(b)
        dst[3 * i] = r
        dst[3 * i + 1] = g
        dst[3 * i + 2] = b


@export("mcs_srgb_transfer")
def mcs_srgb_transfer(src_addr: Int, dst_addr: Int, n: Int, encode: Int) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        dst[i] = srgb_encode(src[i]) if encode != 0 else srgb_decode(src[i])


def xyz_to_lab_range(
    src: Ptr,
    dst: Ptr,
    start: Int,
    end: Int,
    xn: Float64,
    yn: Float64,
    zn: Float64,
) -> None:
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var simd_end = end - (end - start) % W
    while i < simd_end:
        var offset = 3 * i
        var fx = lab_f_simd((src + offset).strided_load[width=W](3) / xn)
        var fy = lab_f_simd((src + offset + 1).strided_load[width=W](3) / yn)
        var fz = lab_f_simd((src + offset + 2).strided_load[width=W](3) / zn)
        (dst + offset).strided_store(116.0 * fy - 16.0, 3)
        (dst + offset + 1).strided_store(500.0 * (fx - fy), 3)
        (dst + offset + 2).strided_store(200.0 * (fy - fz), 3)
        i += W
    while i < end:
        var fx = lab_f(src[3 * i] / xn)
        var fy = lab_f(src[3 * i + 1] / yn)
        var fz = lab_f(src[3 * i + 2] / zn)
        dst[3 * i] = 116.0 * fy - 16.0
        dst[3 * i + 1] = 500.0 * (fx - fy)
        dst[3 * i + 2] = 200.0 * (fy - fz)
        i += 1


@export("mcs_xyz_to_lab")
def mcs_xyz_to_lab(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    xn: Float64,
    yn: Float64,
    zn: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    if n < PARALLEL_THRESHOLD:
        xyz_to_lab_range(src, dst, 0, n, xn, yn, zn)
        return

    var num_tasks = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK

    def worker(
        task: Int,
    ) {imm src, imm dst, imm n, imm xn, imm yn, imm zn}:
        var start = task * PARALLEL_CHUNK
        var end = min(start + PARALLEL_CHUNK, n)
        xyz_to_lab_range(src, dst, start, end, xn, yn, zn)

    parallelize(worker, num_tasks, min(num_tasks, 16))


@export("mcs_lab_to_xyz")
def mcs_lab_to_xyz(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    xn: Float64,
    yn: Float64,
    zn: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var fy = (src[3 * i] + 16.0) / 116.0
        var fx = src[3 * i + 1] / 500.0 + fy
        var fz = fy - src[3 * i + 2] / 200.0
        dst[3 * i] = xn * lab_f_inverse(fx)
        dst[3 * i + 1] = yn * lab_f_inverse(fy)
        dst[3 * i + 2] = zn * lab_f_inverse(fz)


@export("mcs_xyz_to_luv")
def mcs_xyz_to_luv(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    xn: Float64,
    yn: Float64,
    zn: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    var wd = xn + 15.0 * yn + 3.0 * zn
    var un = 4.0 * xn / wd
    var vn = 9.0 * yn / wd
    for i in range(n):
        var x = src[3 * i]
        var y = src[3 * i + 1]
        var z = src[3 * i + 2]
        var yr = y / yn
        var l = 116.0 * pow(yr, 1.0 / 3.0) - 16.0 if yr > 216.0 / 24389.0 else (24389.0 / 27.0) * yr
        var d = x + 15.0 * y + 3.0 * z
        var up = 0.0
        var vp = 0.0
        if d != 0.0:
            up = 4.0 * x / d
            vp = 9.0 * y / d
        dst[3 * i] = l
        dst[3 * i + 1] = 13.0 * l * (up - un)
        dst[3 * i + 2] = 13.0 * l * (vp - vn)


@export("mcs_luv_to_xyz")
def mcs_luv_to_xyz(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    xn: Float64,
    yn: Float64,
    zn: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    var wd = xn + 15.0 * yn + 3.0 * zn
    var un = 4.0 * xn / wd
    var vn = 9.0 * yn / wd
    for i in range(n):
        var l = src[3 * i]
        if l == 0.0:
            dst[3 * i] = 0.0
            dst[3 * i + 1] = 0.0
            dst[3 * i + 2] = 0.0
        else:
            var fy = (l + 16.0) / 116.0
            var y = yn * (fy * fy * fy if l > 8.0 else l * 27.0 / 24389.0)
            var up = src[3 * i + 1] / (13.0 * l) + un
            var vp = src[3 * i + 2] / (13.0 * l) + vn
            var x = 9.0 * y * up / (4.0 * vp)
            var z = y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp)
            dst[3 * i] = x
            dst[3 * i + 1] = y
            dst[3 * i + 2] = z


@export("mcs_xyz_xyy")
def mcs_xyz_xyy(
    src_addr: Int, dst_addr: Int, n: Int, inverse: Int, ix: Float64, iy: Float64
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        if inverse == 0:
            var x = src[3 * i]
            var y = src[3 * i + 1]
            var z = src[3 * i + 2]
            var total = x + y + z
            if total == 0.0:
                dst[3 * i] = 0.0
                dst[3 * i + 1] = 0.0
            else:
                dst[3 * i] = x / total
                dst[3 * i + 1] = y / total
            dst[3 * i + 2] = y
        else:
            var cx = src[3 * i]
            var cy = src[3 * i + 1]
            var lum = src[3 * i + 2]
            if cy == 0.0:
                dst[3 * i] = 0.0
                dst[3 * i + 1] = 0.0
                dst[3 * i + 2] = 0.0
            else:
                dst[3 * i] = cx * lum / cy
                dst[3 * i + 1] = lum
                dst[3 * i + 2] = (1.0 - cx - cy) * lum / cy


def oklab_forward(x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    var l = spow(0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z, 1.0 / 3.0)
    var m = spow(0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z, 1.0 / 3.0)
    var s = spow(0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z, 1.0 / 3.0)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklab_inverse(lab0: Float64, lab1: Float64, lab2: Float64) -> Tuple[Float64, Float64, Float64]:
    var l = lab0 + 0.3963377774 * lab1 + 0.2158037573 * lab2
    var m = lab0 - 0.1055613458 * lab1 - 0.0638541728 * lab2
    var s = lab0 - 0.0894841775 * lab1 - 1.2914855480 * lab2
    l *= l * l
    m *= m * m
    s *= s * s
    return (
        1.2270138511 * l - 0.5577999807 * m + 0.2812561490 * s,
        -0.0405801784 * l + 1.1122568696 * m - 0.0716766787 * s,
        -0.0763812845 * l - 0.4214819784 * m + 1.5861632204 * s,
    )


def ipt_forward(x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    var l = spow(0.4002 * x + 0.7075 * y - 0.0807 * z, 0.43)
    var m = spow(-0.2280 * x + 1.1500 * y + 0.0612 * z, 0.43)
    var s = spow(0.9184 * z, 0.43)
    return (
        0.4 * l + 0.4 * m + 0.2 * s,
        4.455 * l - 4.851 * m + 0.396 * s,
        0.8056 * l + 0.3572 * m - 1.1628 * s,
    )


def ipt_inverse(i0: Float64, p0: Float64, t0: Float64) -> Tuple[Float64, Float64, Float64]:
    var l = spow(i0 + 0.09756893 * p0 + 0.20522643 * t0, 1.0 / 0.43)
    var m = spow(i0 - 0.11387649 * p0 + 0.13321716 * t0, 1.0 / 0.43)
    var s = spow(i0 + 0.03261511 * p0 - 0.67688718 * t0, 1.0 / 0.43)
    return (
        1.85024294 * l - 1.13830164 * m + 0.23843496 * s,
        0.36683078 * l + 0.64388454 * m - 0.01067344 * s,
        1.08885017 * s,
    )


@export("mcs_perceptual")
def mcs_perceptual(src_addr: Int, dst_addr: Int, n: Int, mode: Int) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var a = src[3 * i]
        var b = src[3 * i + 1]
        var c = src[3 * i + 2]
        var v: Tuple[Float64, Float64, Float64]
        if mode == 0:
            v = oklab_forward(a, b, c)
        elif mode == 1:
            v = oklab_inverse(a, b, c)
        elif mode == 2:
            v = ipt_forward(a, b, c)
        else:
            v = ipt_inverse(a, b, c)
        dst[3 * i] = v[0]
        dst[3 * i + 1] = v[1]
        dst[3 * i + 2] = v[2]


def pq_inverse(v: Float64) -> Float64:
    var vp = spow(v / 10000.0, 0.1593017578125)
    return spow(
        (0.8359375 + 18.8515625 * vp) / (1.0 + 18.6875 * vp),
        134.034375,
    )


def pq_forward(v: Float64) -> Float64:
    var vp = spow(v, 1.0 / 134.034375)
    var numerator = max(0.0, vp - 0.8359375)
    return 10000.0 * spow(numerator / (18.8515625 - 18.6875 * vp), 1.0 / 0.1593017578125)


def jz_forward(x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    var xp = 1.15 * x - 0.15 * z
    var yp = 0.66 * y + 0.34 * x
    var l = pq_inverse(0.41478972 * xp + 0.579999 * yp + 0.014648 * z)
    var m = pq_inverse(-0.20151 * xp + 1.120649 * yp + 0.0531008 * z)
    var s = pq_inverse(-0.0166008 * xp + 0.2648 * yp + 0.6684799 * z)
    var iz = 0.5 * l + 0.5 * m
    var az = 3.524 * l - 4.066708 * m + 0.542708 * s
    var bz = 0.199076 * l + 1.096799 * m - 1.295875 * s
    var jz = 0.44 * iz / (1.0 - 0.56 * iz) - 1.6295499532821565e-11
    return (jz, az, bz)


def jz_inverse(jz: Float64, az: Float64, bz: Float64) -> Tuple[Float64, Float64, Float64]:
    var iz = (jz + 1.6295499532821565e-11) / (
        0.44 + 0.56 * (jz + 1.6295499532821565e-11)
    )
    var lp = iz + 0.1386050432715393 * az + 0.0580473161561189 * bz
    var mp = iz - 0.1386050432715393 * az - 0.0580473161561189 * bz
    var sp = iz - 0.0960192420263189 * az - 0.8118918960560390 * bz
    var l = pq_forward(lp)
    var m = pq_forward(mp)
    var s = pq_forward(sp)
    var xp = 1.9242264358 * l - 1.0047923126 * m + 0.0376514040 * s
    var yp = 0.3503167621 * l + 0.7264811939 * m - 0.0653844229 * s
    var z = -0.0909828109 * l - 0.3127282905 * m + 1.5227665613 * s
    var x = (xp + 0.15 * z) / 1.15
    var y = (yp - 0.34 * x) / 0.66
    return (x, y, z)


def jzazbz_range(
    src: Ptr, dst: Ptr, start: Int, end: Int, inverse: Int
) -> None:
    for i in range(start, end):
        var v = (
            jz_inverse(src[3 * i], src[3 * i + 1], src[3 * i + 2])
            if inverse != 0
            else jz_forward(src[3 * i], src[3 * i + 1], src[3 * i + 2])
        )
        dst[3 * i] = v[0]
        dst[3 * i + 1] = v[1]
        dst[3 * i + 2] = v[2]


def jzazbz_gpu_kernel(src: Ptr, dst: Ptr, n: Int, inverse: Int):
    var i = Int(global_idx.x)
    if i < n:
        var v = (
            jz_inverse(src[3 * i], src[3 * i + 1], src[3 * i + 2])
            if inverse != 0
            else jz_forward(src[3 * i], src[3 * i + 1], src[3 * i + 2])
        )
        dst[3 * i] = v[0]
        dst[3 * i + 1] = v[1]
        dst[3 * i + 2] = v[2]


@export("mcs_jzazbz")
def mcs_jzazbz(src_addr: Int, dst_addr: Int, n: Int, inverse: Int) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    if n < PARALLEL_THRESHOLD:
        jzazbz_range(src, dst, 0, n, inverse)
        return

    var num_tasks = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK

    def worker(
        task: Int,
    ) {imm src, imm dst, imm n, imm inverse}:
        var start = task * PARALLEL_CHUNK
        var end = min(start + PARALLEL_CHUNK, n)
        jzazbz_range(src, dst, start, end, inverse)

    parallelize(worker, num_tasks, min(num_tasks, 16))


@export("mcs_jzazbz_gpu")
def mcs_jzazbz_gpu(
    src_addr: Int, dst_addr: Int, n: Int, inverse: Int
) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var memory = ctx.get_memory_info()
        var values = 3 * n
        var allocation_bytes = UInt(values) * UInt(16)
        if memory[0] < UInt(4000 * 1024 * 1024):
            return 0
        if allocation_bytes > UInt(2 * 1024 * 1024 * 1024):
            return 0
        var device_src = ctx.enqueue_create_buffer[DType.float64](values)
        var device_dst = ctx.enqueue_create_buffer[DType.float64](values)
        var src = p(src_addr)
        var dst = p(dst_addr)
        ctx.enqueue_copy(device_src, src)
        comptime block_size = 256
        var grid_size = (n + block_size - 1) // block_size
        ctx.enqueue_function[jzazbz_gpu_kernel](
            device_src,
            device_dst,
            n,
            inverse,
            grid_dim=grid_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(dst, device_dst)
        ctx.synchronize()
        return 1
    except:
        return 0


def hue_degrees(y: Float64, x: Float64) -> Float64:
    var h = atan2(y, x) * 180.0 / PI
    if h < 0.0:
        h += 360.0
    return h


def delta_e_2000(
    l1: Float64,
    a1: Float64,
    b1: Float64,
    l2: Float64,
    a2: Float64,
    b2: Float64,
    textiles: Bool,
) -> Float64:
    var c1ab = sqrt(a1 * a1 + b1 * b1)
    var c2ab = sqrt(a2 * a2 + b2 * b2)
    var cbar = (c1ab + c2ab) / 2.0
    var cbar7 = pow(cbar, 7.0)
    var g = 0.5 * (1.0 - sqrt(cbar7 / (cbar7 + 6103515625.0)))
    var ap1 = (1.0 + g) * a1
    var ap2 = (1.0 + g) * a2
    var cp1 = sqrt(ap1 * ap1 + b1 * b1)
    var cp2 = sqrt(ap2 * ap2 + b2 * b2)
    var hp1 = 0.0 if cp1 == 0.0 else hue_degrees(b1, ap1)
    var hp2 = 0.0 if cp2 == 0.0 else hue_degrees(b2, ap2)
    var dl = l2 - l1
    var dc = cp2 - cp1
    var dh = hp2 - hp1
    if cp1 * cp2 == 0.0:
        dh = 0.0
    elif dh > 180.0:
        dh -= 360.0
    elif dh < -180.0:
        dh += 360.0
    var d_big_h = 2.0 * sqrt(cp1 * cp2) * sin(dh * PI / 360.0)
    var lbar = (l1 + l2) / 2.0
    var cpbar = (cp1 + cp2) / 2.0
    var hsum = hp1 + hp2
    var hdiff = abs(hp1 - hp2)
    var hpbar = hsum
    if cp1 * cp2 != 0.0:
        if hdiff <= 180.0:
            hpbar = hsum / 2.0
        elif hsum < 360.0:
            hpbar = (hsum + 360.0) / 2.0
        else:
            hpbar = (hsum - 360.0) / 2.0
    var hr = hpbar * PI / 180.0
    var t = (
        1.0
        - 0.17 * cos(hr - PI / 6.0)
        + 0.24 * cos(2.0 * hr)
        + 0.32 * cos(3.0 * hr + PI / 30.0)
        - 0.20 * cos(4.0 * hr - 63.0 * PI / 180.0)
    )
    var theta = 30.0 * exp(-pow((hpbar - 275.0) / 25.0, 2.0))
    var cpbar7 = pow(cpbar, 7.0)
    var rc = 2.0 * sqrt(cpbar7 / (cpbar7 + 6103515625.0))
    var lterm = (lbar - 50.0) * (lbar - 50.0)
    var sl = 1.0 + 0.015 * lterm / sqrt(20.0 + lterm)
    var sc = 1.0 + 0.045 * cpbar
    var sh = 1.0 + 0.015 * cpbar * t
    var rt = -sin(2.0 * theta * PI / 180.0) * rc
    var kl = 2.0 if textiles else 1.0
    var vl = dl / (kl * sl)
    var vc = dc / sc
    var vh = d_big_h / sh
    return sqrt(vl * vl + vc * vc + vh * vh + rt * vc * vh)


@export("mcs_delta_e")
def mcs_delta_e(
    a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, method: Int, textiles: Int
) abi("C"):
    var a = p(a_addr)
    var b = p(b_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var l1 = a[3 * i]
        var aa1 = a[3 * i + 1]
        var bb1 = a[3 * i + 2]
        var l2 = b[3 * i]
        var aa2 = b[3 * i + 1]
        var bb2 = b[3 * i + 2]
        if method == 0:
            var dl = l1 - l2
            var da = aa1 - aa2
            var db = bb1 - bb2
            dst[i] = sqrt(dl * dl + da * da + db * db)
        elif method == 1:
            var c1 = sqrt(aa1 * aa1 + bb1 * bb1)
            var c2 = sqrt(aa2 * aa2 + bb2 * bb2)
            var dl = l1 - l2
            var dc = c1 - c2
            var radical = max(0.0, (aa1 - aa2) ** 2 + (bb1 - bb2) ** 2 - dc * dc)
            var dh = sqrt(radical)
            var k1 = 0.048 if textiles != 0 else 0.045
            var k2 = 0.014 if textiles != 0 else 0.015
            var kl = 2.0 if textiles != 0 else 1.0
            dst[i] = sqrt(
                (dl / kl) ** 2
                + (dc / (1.0 + k1 * c1)) ** 2
                + (dh / (1.0 + k2 * c1)) ** 2
            )
        else:
            dst[i] = delta_e_2000(
                l1, aa1, bb1, l2, aa2, bb2, textiles != 0
            )


def response_compress(v: Float64, fl: Float64) -> Float64:
    var x = spow(fl * abs(v) / 100.0, 0.42)
    return 400.0 * sign(v) * x / (27.13 + x) + 0.1


def response_inverse(v: Float64, fl: Float64) -> Float64:
    var x = abs(v - 0.1)
    return sign(v - 0.1) * 100.0 / fl * spow(27.13 * x / (400.0 - x), 1.0 / 0.42)


def appearance_forward_range(
    src: Ptr,
    dst: Ptr,
    matrix: Ptr,
    start: Int,
    end: Int,
    fl: Float64,
    background_n: Float64,
    nbb: Float64,
    zbase: Float64,
    surround_c: Float64,
    surround_nc: Float64,
    aw: Float64,
) -> None:
    var ncb = nbb
    var fl_quarter = pow(fl, 0.25)
    var chroma_factor = pow(1.64 - pow(0.29, background_n), 0.73)
    for i in range(start, end):
        var rgb = mat3(matrix, src[3 * i], src[3 * i + 1], src[3 * i + 2])
        var r = response_compress(rgb[0], fl)
        var g = response_compress(rgb[1], fl)
        var b = response_compress(rgb[2], fl)
        var ca = r - 12.0 * g / 11.0 + b / 11.0
        var cb = (r + g - 2.0 * b) / 9.0
        var h = hue_degrees(cb, ca)
        var et = 0.25 * (cos(2.0 + h * PI / 180.0) + 3.8)
        var achromatic = (2.0 * r + g + b / 20.0 - 0.305) * nbb
        var j = 100.0 * spow(achromatic / aw, surround_c * zbase)
        var q = 4.0 / surround_c * sqrt(j / 100.0) * (aw + 4.0) * fl_quarter
        var t = (
            (50000.0 / 13.0)
            * surround_nc
            * ncb
            * et
            * sqrt(ca * ca + cb * cb)
            / (r + g + 21.0 * b / 20.0)
        )
        var chroma = spow(t, 0.9) * sqrt(j / 100.0) * chroma_factor
        var colourfulness = chroma * fl_quarter
        var saturation = 100.0 * sqrt(colourfulness / q)
        dst[6 * i] = j
        dst[6 * i + 1] = chroma
        dst[6 * i + 2] = h
        dst[6 * i + 3] = saturation
        dst[6 * i + 4] = q
        dst[6 * i + 5] = colourfulness


@export("mcs_appearance_forward")
def mcs_appearance_forward(
    src_addr: Int,
    dst_addr: Int,
    matrix_addr: Int,
    n_samples: Int,
    fl: Float64,
    background_n: Float64,
    nbb: Float64,
    zbase: Float64,
    surround_c: Float64,
    surround_nc: Float64,
    aw: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    var matrix = p(matrix_addr)
    if n_samples < PARALLEL_THRESHOLD:
        appearance_forward_range(
            src,
            dst,
            matrix,
            0,
            n_samples,
            fl,
            background_n,
            nbb,
            zbase,
            surround_c,
            surround_nc,
            aw,
        )
        return

    var num_tasks = (
        n_samples + PARALLEL_CHUNK - 1
    ) // PARALLEL_CHUNK

    def worker(
        task: Int,
    ) {
        imm src,
        imm dst,
        imm matrix,
        imm n_samples,
        imm fl,
        imm background_n,
        imm nbb,
        imm zbase,
        imm surround_c,
        imm surround_nc,
        imm aw,
    }:
        var start = task * PARALLEL_CHUNK
        var end = min(start + PARALLEL_CHUNK, n_samples)
        appearance_forward_range(
            src,
            dst,
            matrix,
            start,
            end,
            fl,
            background_n,
            nbb,
            zbase,
            surround_c,
            surround_nc,
            aw,
        )

    parallelize(worker, num_tasks, min(num_tasks, 16))


@export("mcs_appearance_inverse")
def mcs_appearance_inverse(
    src_addr: Int,
    dst_addr: Int,
    inverse_matrix_addr: Int,
    n_samples: Int,
    fl: Float64,
    background_n: Float64,
    nbb: Float64,
    zbase: Float64,
    surround_c: Float64,
    surround_nc: Float64,
    aw: Float64,
) abi("C"):
    var src = p(src_addr)
    var dst = p(dst_addr)
    var inverse_matrix = p(inverse_matrix_addr)
    var chroma_factor = pow(1.64 - pow(0.29, background_n), 0.73)
    for i in range(n_samples):
        var j = src[3 * i]
        var chroma = src[3 * i + 1]
        var h = src[3 * i + 2]
        var j_safe = max(j, 2.220446049250313e-16)
        var t = spow(chroma / (sqrt(j_safe / 100.0) * chroma_factor), 1.0 / 0.9)
        var et = 0.25 * (cos(2.0 + h * PI / 180.0) + 3.8)
        var achromatic = aw * spow(j / 100.0, 1.0 / (surround_c * zbase))
        var p1 = (50000.0 / 13.0) * surround_nc * nbb * et / t
        var p2 = achromatic / nbb + 0.305
        var p3 = 21.0 / 20.0
        var hr = h * PI / 180.0
        var shr = sin(hr)
        var chr = cos(hr)
        var ca: Float64
        var cb: Float64
        var numerator = p2 * (2.0 + p3) * (460.0 / 1403.0)
        if abs(shr) >= abs(chr):
            cb = numerator / (
                p1 / shr
                + (2.0 + p3) * (220.0 / 1403.0) * chr / shr
                - 27.0 / 1403.0
                + p3 * 6300.0 / 1403.0
            )
            ca = cb * chr / shr
        else:
            ca = numerator / (
                p1 / chr
                + (2.0 + p3) * (220.0 / 1403.0)
                - (27.0 / 1403.0 - p3 * 6300.0 / 1403.0) * shr / chr
            )
            cb = ca * shr / chr
        if t == 0.0:
            ca = 0.0
            cb = 0.0
        var r_a = (460.0 * p2 + 451.0 * ca + 288.0 * cb) / 1403.0
        var g_a = (460.0 * p2 - 891.0 * ca - 261.0 * cb) / 1403.0
        var b_a = (460.0 * p2 - 220.0 * ca - 6300.0 * cb) / 1403.0
        var xyz = mat3(
            inverse_matrix,
            response_inverse(r_a, fl),
            response_inverse(g_a, fl),
            response_inverse(b_a, fl),
        )
        dst[3 * i] = xyz[0]
        dst[3 * i + 1] = xyz[1]
        dst[3 * i + 2] = xyz[2]
