#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline uint dr3_u32(device const uchar* b, ulong o) { return uint(b[o]) | uint(b[o+1])<<8 | uint(b[o+2])<<16 | uint(b[o+3])<<24; }
inline float dr3_float(device const uchar* b, ulong o, ulong i, ulong t) {
    if(t==0) return as_type<float>(dr3_u32(b,o+i*4));
    o+=i*2; ushort v=ushort(b[o])|ushort(b[o+1])<<8;
    return t==1 ? as_type<float>(uint(v)<<16) : float(as_type<half>(v));
}
// Used only when the mapped BF16 tensor starts on a two-byte boundary.
inline float dr3_bf16_aligned(device const uchar* b, ulong o, ulong i) {
    ushort v=*(device const ushort*)(b+o+i*2);
    return as_type<float>(uint(v)<<16);
}
inline float dr3_sigmoid(float v) { return v>=0 ? 1.0f/(1.0f+exp(-v)) : exp(v)/(1.0f+exp(v)); }
kernel void dr3_finite(device const float* x [[buffer(0)]],device atomic_uint* status [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]&&!isfinite(x[i]))atomic_store_explicit(status,1u,memory_order_relaxed);}
kernel void dr3_fill(device float* y [[buffer(0)]],constant ulong* p [[buffer(1)]],uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=as_type<float>(uint(p[1])); }
kernel void dr3_copy(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=x[i]; }
kernel void dr3_convert(device const uchar* x [[buffer(0)]],device float* y [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=dr3_float(x,p[1],i,p[2]); }
kernel void dr3_binary(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=p[1] ? a[i]*b[i] : a[i]+b[i]; }
kernel void dr3_silu(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=x[i]*dr3_sigmoid(x[i]); }
kernel void dr3_rms(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint h [[thread_position_in_grid]]) {
    uint width=uint(p[1]); if(h>=p[0]/width)return; float sum=0;
    for(uint j=0;j<width;j++)sum+=x[h*width+j]*x[h*width+j];
    float inv=1.0f/sqrt(sum/float(width)+as_type<float>(uint(p[2])));
    for(uint j=0;j<width;j++)y[h*width+j]=x[h*width+j]*inv*(p[3]?1.0f+w[j]:w[j]);
}
kernel void dr3_rope(device float* x [[buffer(0)]],constant ulong* p [[buffer(1)]],uint id [[thread_position_in_grid]]) {
    uint pairs=uint(p[2])/2;if(id>=p[0]*pairs)return;uint h=id/pairs,i=id%pairs;
    float angle=float(p[3])/pow(as_type<float>(uint(p[4])),float(2*i)/float(p[2]));float c=cos(angle),s=sin(angle);
    uint a=h*uint(p[1])+(p[5]?2*i:i),b=h*uint(p[1])+(p[5]?2*i+1:i+pairs);float u=x[a],v=x[b];x[a]=u*c-v*s;x[b]=u*s+v*c;
}
kernel void dr3_dense(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float parts[64];uint cols=uint(p[1]);float s=0;ulong row=p[4]*p[0]+r;
    for(uint j=lane;j<cols;j+=64)s+=dr3_float(w,p[2],row*cols+j,p[3])*x[j];parts[lane]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=32;d>0;d/=2){if(lane<d)parts[lane]+=parts[lane+d];threadgroup_barrier(mem_flags::mem_threadgroup);}if(lane==0)y[r]=parts[0];
}
kernel void dr3_dense_four_tree(
    device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]],
    device const uchar* w2 [[buffer(2)]], device const uchar* w3 [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* y0 [[buffer(5)]], device float* y1 [[buffer(6)]],
    device float* y2 [[buffer(7)]], device float* y3 [[buffer(8)]],
    constant ulong* p [[buffer(9)]],
    uint output_row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint projection = 0, r = output_row;
    while (projection < 3 && r >= p[projection * 5]) {
        r -= uint(p[projection * 5]);
        projection++;
    }
    constant ulong* q = p + projection * 5;
    device const uchar* w = projection == 0 ? w0 : projection == 1 ? w1 : projection == 2 ? w2 : w3;
    device float* y = projection == 0 ? y0 : projection == 1 ? y1 : projection == 2 ? y2 : y3;
    threadgroup float parts[64];
    uint cols = uint(q[1]);
    float s = 0;
    ulong row = q[4] * q[0] + r;
    for (uint j = lane; j < cols; j += 64) s += dr3_float(w, q[2], row * cols + j, q[3]) * x[j];
    parts[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 32; d > 0; d /= 2) {
        if (lane < d) parts[lane] += parts[lane + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) y[r] = parts[0];
}
kernel void dr3_dense_pair_tree(
    device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y0 [[buffer(3)]], device float* y1 [[buffer(4)]],
    constant ulong* p [[buffer(5)]],
    uint output_row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint projection = output_row < p[0] ? 0 : 1;
    uint r = projection == 0 ? output_row : output_row - uint(p[0]);
    constant ulong* q = p + projection * 5;
    device const uchar* w = projection == 0 ? w0 : w1;
    device float* y = projection == 0 ? y0 : y1;
    threadgroup float parts[64];
    uint cols = uint(q[1]);
    float s = 0;
    ulong row = q[4] * q[0] + r;
    for (uint j = lane; j < cols; j += 64) s += dr3_float(w, q[2], row * cols + j, q[3]) * x[j];
    parts[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 32; d > 0; d /= 2) {
        if (lane < d) parts[lane] += parts[lane + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) y[r] = parts[0];
}

// Equal-row BF16 specialization: one input load feeds both original F32
// accumulators. The host selects this only for even, in-range weight offsets.
kernel void dr3_dense_pair_bf16_aligned(
    device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y0 [[buffer(3)]], device float* y1 [[buffer(4)]],
    constant ulong* p [[buffer(5)]],
    uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint cols = uint(p[1]);
    ulong row_base = ulong(r) * ulong(cols);
    float s0 = 0.0f, s1 = 0.0f;
    for (uint j = lane; j < cols; j += 64) {
        float xval = x[j];
        ulong element = row_base + ulong(j);
        s0 += dr3_bf16_aligned(w0, p[2], element) * xval;
        s1 += dr3_bf16_aligned(w1, p[7], element) * xval;
    }
    threadgroup float parts0[64], parts1[64];
    parts0[lane] = s0;
    parts1[lane] = s1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 32; d > 0; d /= 2) {
        if (lane < d) {
            parts0[lane] += parts0[lane + d];
            parts1[lane] += parts1[lane + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        y0[r] = parts0[0];
        y1[r] = parts1[0];
    }
}

kernel void dr3_dense_simd(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]],uint simd_lane [[thread_index_in_simdgroup]],uint simdgroup [[simdgroup_index_in_threadgroup]],uint simd_width [[threads_per_simdgroup]]) {
    threadgroup float parts[64];uint cols=uint(p[1]);float s=0;ulong row=p[4]*p[0]+r;
    if(p[3]==1&&(p[2]&1)==0){for(uint j=lane;j<cols;j+=64)s+=dr3_bf16_aligned(w,p[2],row*cols+j)*x[j];}
    else{for(uint j=lane;j<cols;j+=64)s+=dr3_float(w,p[2],row*cols+j,p[3])*x[j];}
    parts[lane]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd_width!=32){
        for(uint d=32;d>0;d/=2){if(lane<d)parts[lane]+=parts[lane+d];threadgroup_barrier(mem_flags::mem_threadgroup);}if(lane==0)y[r]=parts[0];
    }else if(simdgroup==0){
        float sum=parts[simd_lane]+parts[simd_lane+32];
        for(uint d=16;d>0;d/=2){float peer=simd_shuffle_down(sum,ushort(d));if(simd_lane<d)sum+=peer;}
        if(simd_lane==0)y[r]=sum;
    }
}
kernel void dr3_dense_simd32(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint simd_lane [[thread_index_in_simdgroup]],uint simd_width [[threads_per_simdgroup]]) {
    if(simd_width!=32)return;
    uint cols=uint(p[1]);ulong row=p[4]*p[0]+r;float lo=0,hi=0;
    if(p[3]==1&&(p[2]&1)==0){
        for(uint j=simd_lane;j<cols;j+=64)lo+=dr3_bf16_aligned(w,p[2],row*cols+j)*x[j];
        for(uint j=simd_lane+32;j<cols;j+=64)hi+=dr3_bf16_aligned(w,p[2],row*cols+j)*x[j];
    }else{
        for(uint j=simd_lane;j<cols;j+=64)lo+=dr3_float(w,p[2],row*cols+j,p[3])*x[j];
        for(uint j=simd_lane+32;j<cols;j+=64)hi+=dr3_float(w,p[2],row*cols+j,p[3])*x[j];
    }
    float sum=lo+hi;
    for(uint d=16;d>0;d/=2){float peer=simd_shuffle_down(sum,ushort(d));if(simd_lane<d)sum+=peer;}
    if(simd_lane==0)y[r]=sum;
}
kernel void dr3_expert(device const uchar* w [[buffer(0)]],device const uchar* s [[buffer(1)]],device const uchar* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* y [[buffer(4)]],device const float* ids [[buffer(5)]],constant ulong* p [[buffer(6)]],threadgroup float* parts [[threadgroup(0)]],uint r [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    float selected=p[10]?ids[p[11]]:float(p[4]);if(!isfinite(selected)||selected<0||selected>=float(p[12])||floor(selected)!=selected){if(lane==0)y[r]=NAN;return;}
    uint rows=uint(p[0]),cols=uint(p[1]),bits=uint(p[2]),group=uint(p[3]),e=uint(selected),row=e*rows+r;
    uint groups=cols/group,per=32/bits,mask=(1u<<bits)-1;
    for(uint g=lane;g<groups;g+=64){uint ix=row*groups+g;float scale=dr3_float(s,p[6],ix,p[8]),bias=dr3_float(b,p[7],ix,p[9]),part=0;
      for(uint j=0;j<group/per;j++){uint col=g*group+j*per;uint word=dr3_u32(w,p[5]+(ulong(row)*(cols/per)+col/per)*4);
        for(uint k=0;k<per;k++){float weight=scale*float((word>>(k*bits))&mask)+bias;part+=weight*x[col+k];}}
      parts[g]=part;}
    threadgroup_barrier(mem_flags::mem_threadgroup);if(lane==0){float sum=0;for(uint g=0;g<groups;g++)sum+=parts[g];y[r]=sum;}
}
kernel void dr3_embedding(device const uchar* w [[buffer(0)]],device const float* token [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint i [[thread_position_in_grid]]) { if(i<p[1]){int r=int(token[0]);y[i]=r>=0 && r<int(p[0])?dr3_float(w,p[2],ulong(r)*p[1]+i,p[3]):NAN;} }
kernel void dr3_experts(
    device const uchar* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const uchar* b [[buffer(2)]], device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]], device const float* ids [[buffer(5)]],
    constant ulong* p [[buffer(6)]], threadgroup float* parts [[threadgroup(0)]],
    uint2 pos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint r = pos.x, rank = pos.y, rows = uint(p[0]), cols = uint(p[1]);
    float selected = ids[rank];
    if (!isfinite(selected) || selected < 0 || selected >= float(p[9]) || floor(selected) != selected) {
        if (lane == 0) y[rank * rows + r] = NAN;
        return;
    }
    uint bits = uint(p[2]), group = uint(p[3]), row = uint(selected) * rows + r;
    uint groups = cols / group, per = 32 / bits, mask = (1u << bits) - 1;
    uint input = p[10] ? rank * cols : 0;
    for (uint g = lane; g < groups; g += 64) {
        uint ix = row * groups + g;
        float scale = dr3_float(s, p[5], ix, p[7]), bias = dr3_float(b, p[6], ix, p[8]), part = 0;
        for (uint j = 0; j < group / per; j++) {
            uint col = g * group + j * per;
            uint word = dr3_u32(w, p[4] + (ulong(row) * (cols / per) + col / per) * 4);
            for (uint k = 0; k < per; k++) {
                float weight = scale * float((word >> (k * bits)) & mask) + bias;
                part += weight * x[input + col + k];
            }
        }
        parts[g] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum = 0;
        for (uint g = 0; g < groups; g++) sum += parts[g];
        y[rank * rows + r] = sum;
    }
}
// Optional batched-only row packing. Each row keeps dr3_experts' packed-word
// loads, affine dequantization and sequential F32 subtotal. The row leader
// then adds its group subtotals in the same ascending order as the native
// kernel. Every lane writes scratch and reaches the barrier, including tail
// rows in the final threadgroup.
kernel void dr3_experts_packed_rows(
    device const uchar* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const uchar* b [[buffer(2)]], device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]], device const float* ids [[buffer(5)]],
    constant ulong* p [[buffer(6)]], threadgroup float* parts [[threadgroup(0)]],
    uint2 pos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint rows = uint(p[0]), cols = uint(p[1]), bits = uint(p[2]), group = uint(p[3]);
    uint groups = cols / group, rows_per_threadgroup = 64 / groups;
    uint row_slot = lane / groups, group_index = lane % groups;
    uint row = pos.x * rows_per_threadgroup + row_slot, rank = pos.y;
    float selected = ids[rank];
    bool valid_expert = isfinite(selected) && selected >= 0 && selected < float(p[9]) && floor(selected) == selected;
    uint per = 32 / bits, mask = (1u << bits) - 1;
    uint input_offset = p[10] ? rank * cols : 0;

    float partial = 0.0f;
    if (row < rows && valid_expert) {
        uint expert_row = uint(selected) * rows + row;
        uint ix = expert_row * groups + group_index;
        float scale = dr3_float(s, p[5], ix, p[7]);
        float bias = dr3_float(b, p[6], ix, p[8]);
        for (uint word_index = 0; word_index < group / per; word_index++) {
            uint column = group_index * group + word_index * per;
            uint word = dr3_u32(w, p[4] + (ulong(expert_row) * (cols / per) + column / per) * 4);
            for (uint value_index = 0; value_index < per; value_index++) {
                float weight = scale * float((word >> (value_index * bits)) & mask) + bias;
                partial += weight * x[input_offset + column + value_index];
            }
        }
    }
    parts[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < rows && group_index == 0) {
        if (!valid_expert) {
            y[rank * rows + row] = NAN;
        } else {
            float sum = 0.0f;
            uint row_base = row_slot * groups;
            for (uint group_slot = 0; group_slot < groups; group_slot++) {
                sum += parts[row_base + group_slot];
            }
            y[rank * rows + row] = sum;
        }
    }
}

kernel void dr3_embedding_quant(device const uchar* w [[buffer(0)]],device const uchar* s [[buffer(1)]],device const uchar* b [[buffer(2)]],device const float* token [[buffer(3)]],device float* y [[buffer(4)]],constant ulong* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1])return;int r=int(token[0]);if(r<0||r>=int(p[0])){y[i]=NAN;return;}uint bits=uint(p[2]),per=32/bits,group=uint(p[3]);ulong ix=ulong(r)*(p[1]/group)+i/group;
    uint word=dr3_u32(w,p[4]+(ulong(r)*(p[1]/per)+i/per)*4);y[i]=dr3_float(s,p[5],ix,p[7])*float((word>>((i%per)*bits))&((1u<<bits)-1))+dr3_float(b,p[6],ix,p[8]);
}
kernel void dr3_argmax(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i)return;uint best=0;for(uint j=1;j<p[0];j++)if(x[j]>x[best])best=j;y[0]=float(best);}
kernel void dr3_dense_selected(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const float* ids [[buffer(2)]],device float* y [[buffer(3)]],constant ulong* p [[buffer(4)]],uint r [[thread_position_in_grid]]) {
 if(r>=p[0])return;float id=ids[p[4]];if(!isfinite(id)||id<0||id>=float(p[5])||floor(id)!=id){y[r]=NAN;return;}ulong row=ulong(id)*p[0]+r;float sum=0;for(uint j=0;j<p[1];j++)sum+=dr3_float(w,p[2],row*p[1]+j,p[3])*x[j];y[r]=sum;
}
kernel void dr3_sample(device const float* x [[buffer(0)]],device const float* draw [[buffer(1)]],device float* y [[buffer(2)]],constant ulong* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i)return;uint n=uint(p[0]),k=uint(p[1]),ids[1024];float vals[1024],temp=as_type<float>(uint(p[2])),top=as_type<float>(uint(p[3]));float maximum=-INFINITY;
    if(!isfinite(draw[0])||draw[0]<0.0f||draw[0]>1.0f){y[0]=NAN;return;}
    for(uint j=0;j<n;j++){if(!isfinite(x[j])){y[0]=NAN;return;}float v=x[j]/temp;maximum=max(maximum,v);uint pos=min(j,k);while(pos>0 && (v>vals[pos-1] || (v==vals[pos-1]&&j<ids[pos-1]))){if(pos<k){vals[pos]=vals[pos-1];ids[pos]=ids[pos-1];}pos--;}if(pos<k){vals[pos]=v;ids[pos]=j;}}
    float denom=0;for(uint j=0;j<n;j++)denom+=exp(x[j]/temp-maximum);
    float cum=0,total=0;uint kept=0;for(uint j=0;j<k;j++){float v=exp(vals[j]-maximum);cum+=v/denom;if(j==0||cum<=top){vals[kept]=v;ids[kept]=ids[j];total+=v;kept++;}}
    float d=draw[0]*total;uint chosen=ids[kept-1];for(uint j=0;j<kept;j++){d-=vals[j];if(d<0){chosen=ids[j];break;}}y[0]=float(chosen);
}
