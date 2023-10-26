kernel void mul(
    constant float4 *src0 [[buffer(0)]],
    constant float4 *src1 [[buffer(1)]],
    device float4 *dst [[buffer(2)]],
    uint tpig [[thread_position_in_grid]])
{
    dst[tpig] = src0[tpig] * src1[tpig];
}

kernel void dot_product(
  constant uint *inA [[buffer(0)]],
  constant uint *inB [[buffer(1)]],
  device uint *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}