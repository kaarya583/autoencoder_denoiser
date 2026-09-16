"""Exact portable C arithmetic against NumPy and independent Torch dots."""
import ctypes as ct

import numpy as np
import pytest
import torch
from torch import nn

from esp32_denoiser.experimental_gtcrn_ops import (
    ActivationGrid as Grid, IntegerAffine, IntegerPReLU, IntegerStreamConv,
    attention_energy, attention_product, requantize_activation, residual_add, shuffle_pair, subband_features,
)
from esp32_denoiser.experimental_native import _buffer, _handle
from esp32_denoiser.experimental_ops_native import CIntegerAffine, CIntegerOps, CIntegerStreamConv
from esp32_denoiser.vendor.gtcrn.convolution import StreamConv2d, StreamConvTranspose2d
from test_experimental_gtcrn_ops import _source, _torch_affine, _requantize_oracle


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("kind", ["linear","grouped","depthwise","transpose","transpose_point"])
@pytest.mark.parametrize("exponents", [(-5,-3),(-8,-6)])
def test_native_affines_exact_raw_dots_and_folded_signed_bn(kind,exponents):
    torch.manual_seed(567)
    layer,shape=_source(kind)
    layer=layer.eval()
    outputs=layer.out_features if kind == "linear" else layer.out_channels
    bn=(nn.BatchNorm1d(outputs) if kind == "linear" else nn.BatchNorm2d(outputs)).double().eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.linspace(-.3,.4,outputs))
        bn.running_var.copy_(torch.linspace(.4,1.3,outputs))
        bn.weight.copy_(torch.linspace(-1.2,.7,outputs))
        bn.bias.copy_(torch.linspace(.01,-.01,outputs))
    snapshot=IntegerAffine.from_torch(layer,Grid(exponents[0]),Grid(exponents[1]),batch_norm=bn)
    native=CIntegerAffine(snapshot)
    x=np.random.default_rng(38).integers(-128,128,shape,dtype=np.int8)
    output,acc=native(x,return_accumulator=True)
    expected=_torch_affine(layer,torch.from_numpy(x.astype(np.float64)),snapshot.weights,snapshot.bias).numpy()
    np.testing.assert_array_equal(acc,expected)
    np.testing.assert_array_equal(output,_requantize_oracle(expected,snapshot.shifts,-1 if kind == "linear" else 1))
    np.testing.assert_array_equal(output,snapshot(x))
    assert acc.dtype == np.int32 and output.dtype == np.int8


@pytest.mark.parametrize("transpose,stride,padding", [(False,1,1),(True,1,1),(True,2,1),(True,2,4)])
def test_streaming_caches_and_upsampling_crop_match_without_extra_kernel_flip(transpose,stride,padding):
    torch.manual_seed(474)
    wrapper=(StreamConvTranspose2d if transpose else StreamConv2d)(
        4,4,(3,3),stride=(1,stride),padding=(0,padding),dilation=(2,1),groups=2).eval()
    snapshot=IntegerStreamConv.from_torch(wrapper,Grid(-4),Grid(-2))
    native=CIntegerStreamConv(snapshot)
    rng=np.random.default_rng(574)
    x=rng.integers(-128,128,(2,4,29,9),dtype=np.int8)
    state=native.initial_state(2,9); reference=state.copy()
    for t in range(x.shape[2]):
        previous=state.copy()
        actual,state=native.step(x[:,:,t:t+1],state)
        expected,reference=snapshot.step(x[:,:,t:t+1],reference)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(state,reference)
        np.testing.assert_array_equal(previous[:,:,-1],x[:,:,t-1] if t else np.zeros((2,4,9),np.int8))
    assert state.dtype == np.int8 and state.nbytes == 2*4*4*9
    # Exercise the C-supported exact in-place persistent history update.
    frame=np.ascontiguousarray(x[:1,:,:1]); old=native.initial_state(1,9)
    expected,expected_state=snapshot.step(frame,old)
    output=np.empty_like(expected); work=np.empty(native.workspace_bytes(9),np.int8)
    assert native.affine.library.ednx_stream_conv(native.affine.handle,frame.ctypes.data,frame.nbytes,
        old.ctypes.data,old.nbytes,9,native.stride,native.left,native.right,output.ctypes.data,output.nbytes,
        old.ctypes.data,old.nbytes,work.ctypes.data,work.nbytes) == 0
    np.testing.assert_array_equal(output,expected); np.testing.assert_array_equal(old,expected_state)


@pytest.mark.parametrize("ie,oe", [(-16,8),(8,-16),(-4,-5),(-5,-4)])
def test_all_signed_codes_regrid_and_single_round_residual(ie,oe):
    ops=CIntegerOps(); values=np.arange(-128,128,dtype=np.int16).astype(np.int8)
    np.testing.assert_array_equal(ops.regrid(values,Grid(ie),Grid(oe)),requantize_activation(values,Grid(ie),Grid(oe)))
    right=values[::-1].copy()
    np.testing.assert_array_equal(ops.residual(values,right,Grid(ie),Grid(oe),Grid(-3)),
                                 residual_add(values,right,Grid(ie),Grid(oe),Grid(-3)))
    np.testing.assert_array_equal(ops.residual(np.array([1,-1],np.int8),np.array([1,-1],np.int8),
                                 Grid(-5),Grid(-5),Grid(-4)),[1,-1])
    np.testing.assert_array_equal(ops.residual(np.array([127],np.int8),np.array([-127],np.int8),
                                 Grid(8),Grid(8),Grid(-16)),[0])
    inplace=values.copy()
    assert ops.library.ednx_regrid(inplace.ctypes.data,inplace.ctypes.data,inplace.size,ie,oe) == 0
    np.testing.assert_array_equal(inplace,ops.regrid(values,Grid(ie),Grid(oe)))


@pytest.mark.parametrize("slopes", [[.25],[-2.25,0,1,2.25]])
@pytest.mark.parametrize("ie,oe", [(-4,-6),(8,-16),(-16,8)])
def test_prelu_negative_zero_and_wide_signed_products(slopes,ie,oe):
    layer=nn.PReLU(len(slopes))
    with torch.no_grad(): layer.weight.copy_(torch.tensor(slopes))
    snapshot=IntegerPReLU(layer,Grid(ie),Grid(oe))
    x=np.arange(-128,128,dtype=np.int16).astype(np.int8)[None,None,None].repeat(4,1)
    np.testing.assert_array_equal(CIntegerOps().prelu(x,snapshot),snapshot(x))


@pytest.mark.parametrize("ie,oe,frequency", [(-4,-7,33),(-16,8,1024),(8,-16,1024),(-5,-6,1)])
def test_attention_energy_and_endpoint_probability_codes(ie,oe,frequency):
    ops=CIntegerOps(); rng=np.random.default_rng(743)
    x=rng.integers(-128,128,(2,8,1,frequency),dtype=np.int8)
    x[:,0]=-128; x[:,1]=0
    p=rng.integers(-128,128,(2,8,1,1),dtype=np.int8); p[:,0]=-128; p[:,1]=127
    np.testing.assert_array_equal(ops.energy(x,Grid(ie),Grid(oe)),attention_energy(x,Grid(ie),Grid(oe)))
    np.testing.assert_array_equal(ops.product(x,p,Grid(ie),Grid(oe)),attention_product(x,p,Grid(ie),Grid(oe)))
    # The native graph uses one /255 probability for all frequencies in a row.
    y=np.empty_like(x); gate=np.ascontiguousarray(p.reshape(-1))
    assert ops.library.ednx_attention_product(x.ctypes.data,gate.ctypes.data,y.ctypes.data,gate.size,frequency,ie,oe) == 0
    np.testing.assert_array_equal(y,attention_product(x,p,Grid(ie),Grid(oe)))
    all_codes=np.arange(-128,128,dtype=np.int16).astype(np.int8)
    np.testing.assert_array_equal(ops.product(all_codes,np.full_like(all_codes,127),Grid(-7),Grid(-7)),all_codes)
    np.testing.assert_array_equal(ops.product(all_codes,np.full_like(all_codes,-128),Grid(-7),Grid(-7)),np.zeros_like(all_codes))


def test_lut_sfe_shuffle_layout_and_signed_endpoints():
    ops=CIntegerOps(); rng=np.random.default_rng(878)
    x=rng.integers(-128,128,(2,8,3,33),dtype=np.int8)
    y=rng.integers(-128,128,x.shape,dtype=np.int8)
    np.testing.assert_array_equal(ops.subband(x),subband_features(x))
    np.testing.assert_array_equal(ops.shuffle(x,y),shuffle_pair(x,y))
    table=np.arange(-128,128,dtype=np.int16).astype(np.int8)[::-1].copy()
    np.testing.assert_array_equal(ops.lut(x,table),table[x.astype(np.int16)+128])


def test_native_rejects_invalid_bounds_and_short_buffers_before_inference():
    layer=nn.Linear(8,3).eval()
    snapshot=IntegerAffine.from_torch(layer,Grid(-4),Grid(-4))
    native=CIntegerAffine(snapshot); lib=native.library
    for mutate in (lambda s:setattr(s,"groups",3),lambda s:setattr(s,"input_exponent",9),
                   lambda s:setattr(s.weights,"bytes",s.weights.bytes-1),lambda s:setattr(s,"kernel_time",0)):
        spec=type(native.spec).from_buffer_copy(native.spec); mutate(spec); handle=_handle(native.handle_bytes)
        assert lib.ednx_affine_init(handle,ct.byref(spec)) == -1
        a,b=ct.c_uint32(),ct.c_uint32()
        assert lib.ednx_affine_output_shape(handle,1,1,ct.byref(a),ct.byref(b)) == -1
    for bias,weight in ((np.full(3,2**31-1,np.int32),np.ones((3,8),np.int8)),
                        (np.full(3,-2**31,np.int32),np.zeros((3,8),np.int8))):
        spec=type(native.spec).from_buffer_copy(native.spec); spec.bias=_buffer(bias); spec.weights=_buffer(weight)
        assert lib.ednx_affine_init(_handle(native.handle_bytes),ct.byref(spec)) == -1
    x=np.zeros((1,8),np.int8); y=np.full((1,3),12,np.int8)
    assert lib.ednx_affine_run(native.handle,x.ctypes.data,7,1,1,y.ctypes.data,y.nbytes,None,0) == -1
    np.testing.assert_array_equal(y,12)
    assert lib.ednx_attention_energy(x.ctypes.data,y.ctypes.data,1,1025,-4,-4) == -1
    assert lib.ednx_prelu(x.ctypes.data,y.ctypes.data,1,3,x.ctypes.data,1,5,-4,-4) == -1


@pytest.mark.parametrize("sign",[-1,1])
def test_affine_near_int32_bound_and_maximum_left_shift(sign):
    snapshot=IntegerAffine.from_torch(nn.Linear(1,1).eval(),Grid(8),Grid(-16))
    snapshot.weights=np.array([[0]],np.int8)
    snapshot.bias=np.array([sign*(2**31-1)],np.int32)
    snapshot.exponents=np.array([4],np.int8)
    snapshot.shifts=np.array([28],np.int64)
    native=CIntegerAffine(snapshot)
    output,acc=native(np.array([[127]],np.int8),return_accumulator=True)
    np.testing.assert_array_equal(acc,[[sign*(2**31-1)]])
    np.testing.assert_array_equal(output,[[127 if sign > 0 else -128]])
