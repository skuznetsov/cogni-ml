#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Private experimental layout. Offsets address the blob, never host pointers.
static inline uint pos_u16(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8);
}
static inline uint pos_u32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
static inline void pos_store16(device uchar *p, uint u) {
    p[0] = uchar(u); p[1] = uchar(u >> 8);
}
static inline float pos_bf16(device const uchar *p) {
    return as_type<float>(pos_u16(p) << 16);
}
static inline float pos_decode(device const uchar *block, device const float *lut,
                               uint within, uint precision) {
    uint code = 0;
    uint byte = 31u - within / 8u, mask = 1u << (within & 7u);
    for (uint p = 0; p < precision; ++p)
        if ((uint(block[8u + p * 32u + byte]) & mask) != 0u) code |= 1u << (7u-p);
    float scaled = as_type<float>(pos_u32(block+4)) * lut[code];
    return as_type<float>(pos_u32(block)) + scaled;
}
static inline bool pos_bad_block(device const uchar *block) {
    float mean = as_type<float>(pos_u32(block));
    float sigma = as_type<float>(pos_u32(block+4));
    return !isfinite(mean) || !isfinite(sigma) || sigma < 0.0f;
}
static inline bool pos_escape(float w, float base, float sigma, float threshold) {
    float error = abs(w-base);
    return sigma == 0.0f ? error != 0.0f : error/sigma > threshold;
}
static inline uint pos_cost(uint tile, uint precision, uint n) {
    uint base = precision == 16u ? tile*2u : tile/256u*(8u+precision*32u);
    uint width = tile == 256u ? 1u : 2u;
    return (4u+base+n*(width+2u)+3u)&~3u;
}
// Parallel group scans below only touch shared memory; no completion boundary
// or device/host transfer is introduced between planning, packing and GEMM.
static inline uint pos_inclusive_scan(threadgroup uint *s, uint t) {
    for (uint stride=1u; stride<256u; stride<<=1u) {
        uint add = t >= stride ? s[t-stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s[t] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[t];
}

kernel void wan_positional_zero(device atomic_uint *status [[buffer(0)]],
                               uint i [[thread_position_in_grid]]) {
    if (i < 8u) atomic_store_explicit(status+i, 0u, memory_order_relaxed);
}

kernel void wan_positional_plan(
    device const uchar *original [[buffer(0)]], device const uchar *p4 [[buffer(1)]],
    device const uchar *p5 [[buffer(2)]], device const float *lut4 [[buffer(3)]],
    device const float *lut5 [[buffer(4)]], device uint4 *plans [[buffer(5)]],
    constant uint &tile [[buffer(6)]], constant float &threshold [[buffer(7)]],
    uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup uint4 counts[256];
    uint4 c = uint4(0u);
    for (uint j=t; j<tile; j+=256u) {
        uint i=g*tile+j, b=i/256u, within=i&255u;
        device const uchar *b4=p4+b*136u, *b5=p5+b*168u;
        float w=pos_bf16(original+i*2u);
        float a=pos_decode(b4,lut4,within,4u), z=pos_decode(b5,lut5,within,5u);
        float sigma=as_type<float>(pos_u32(b4+4));
        c.x+=uint(pos_escape(w,a,sigma,threshold));
        c.y+=uint(pos_escape(w,z,sigma,threshold));
        c.z|=uint(!isfinite(w)||!isfinite(a)||!isfinite(z)||pos_bad_block(b4)||pos_bad_block(b5));
        // Candidate moments must come from the same normalization groups.
        c.z|=uint(pos_u32(b4)!=pos_u32(b5)||pos_u32(b4+4)!=pos_u32(b5+4));
    }
    c.z|=uint(!isfinite(lut4[t])||!isfinite(lut5[t]));
    counts[t]=c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
        if (t<s) {counts[t].xy+=counts[t+s].xy; counts[t].z|=counts[t+s].z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) {
        uint precision=16u, n=0u, bytes=pos_cost(tile,16u,0u);
        uint cost5=pos_cost(tile,5u,counts[0].y), cost4=pos_cost(tile,4u,counts[0].x);
        // Strict saving against the next viable precision; ties keep the
        // higher tier and avoid a sidecar with no byte benefit.
        if (cost5<bytes) {precision=5u; n=counts[0].y; bytes=cost5;}
        if (cost4<bytes) {precision=4u; n=counts[0].x; bytes=cost4;}
        plans[g]=uint4(bytes,precision,n,counts[0].z);
    }
}

kernel void wan_positional_scan(device const uint4 *plans [[buffer(0)]],
    device uint *offsets [[buffer(1)]], device uint *groups [[buffer(2)]],
    constant uint &tiles [[buffer(3)]], uint i [[thread_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup uint sums[256];
    uint value=i<tiles ? plans[i].x : 0u;
    sums[t]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint inclusive=pos_inclusive_scan(sums,t);
    if (i<tiles) offsets[i]=inclusive-value;
    if (t==255u) groups[g]=inclusive;
}

kernel void wan_positional_scan_groups(device uint *groups [[buffer(0)]],
    device uint *offsets [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
    constant uint &count [[buffer(3)]], constant uint &tiles [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i!=0u) return;
    uint total=0u;
    // At most 32768 scalar entries under the host UInt32-size admission.
    for (uint g=0u;g<count;++g) {uint n=groups[g];groups[g]=total;total+=n;}
    offsets[tiles]=total;
    atomic_store_explicit(status,total,memory_order_relaxed);
}

kernel void wan_positional_scan_finish(device const uint *groups [[buffer(0)]],
    device uint *offsets [[buffer(1)]], constant uint &tiles [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i<tiles) offsets[i]+=groups[i/256u];
}

kernel void wan_positional_pack(
    device const uchar *original [[buffer(0)]], device const uchar *p4 [[buffer(1)]],
    device const uchar *p5 [[buffer(2)]], device const float *lut4 [[buffer(3)]],
    device const float *lut5 [[buffer(4)]], device const uint4 *plans [[buffer(5)]],
    device const uint *offsets [[buffer(6)]], device uchar *blob [[buffer(7)]],
    device atomic_uint *status [[buffer(8)]], constant uint &tile [[buffer(9)]],
    constant float &threshold [[buffer(10)]], constant uint &mode [[buffer(11)]],
    constant uint &capacity [[buffer(12)]], uint t [[thread_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]) {
    uint4 plan=plans[g]; uint start=offsets[g], end=offsets[g+1u];
    if (plan.w!=0u) {
        if (t==0u) atomic_fetch_add_explicit(status+5,1u,memory_order_relaxed);
        return;
    }
    if (start>capacity || end>capacity || start>end || end-start!=plan.x) {
        if (t==0u) atomic_fetch_add_explicit(status+6,1u,memory_order_relaxed);
        return;
    }
    uint precision=plan.y, n=plan.z, width=tile==256u ? 1u:2u;
    uint base_bytes=precision==16u ? tile*2u : tile/256u*(8u+precision*32u);
    device uchar *record=blob+start;
    if (t==0u) {
        record[0]=uchar(precision);record[1]=uchar(mode);pos_store16(record+2,n);
        atomic_fetch_add_explicit(status+(precision==4u?1u:precision==5u?2u:3u),1u,memory_order_relaxed);
        atomic_fetch_add_explicit(status+4,n,memory_order_relaxed);
    }
    device const uchar *source=precision==16u ? original+g*tile*2u :
        (precision==4u ? p4+g*(tile/256u)*136u : p5+g*(tile/256u)*168u);
    for (uint j=t;j<base_bytes;j+=256u) record[4u+j]=source[j];
    uint unpadded=4u+base_bytes+n*(width+2u);
    for (uint j=unpadded+t;j<plan.x;j+=256u) record[j]=0u;
    if (precision==16u) return;
    device const float *lut=precision==4u ? lut4:lut5;
    uint stride=8u+precision*32u;
    threadgroup uint ranks[256];
    threadgroup uint previous;
    if (t==0u) previous=0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pass=0u;pass<tile/256u;++pass) {
        uint j=pass*256u+t, i=g*tile+j;
        device const uchar *block=source+pass*stride;
        float w=pos_bf16(original+i*2u), base=pos_decode(block,lut,t,precision);
        float sigma=as_type<float>(pos_u32(block+4));
        uint selected=uint(pos_escape(w,base,sigma,threshold));
        ranks[t]=selected;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint rank=pos_inclusive_scan(ranks,t);
        if (selected!=0u) {
            uint index=previous+rank-1u;
            device uchar *entry=record+4u+base_bytes+index*(width+2u);
            entry[0]=uchar(j);if (width==2u) entry[1]=uchar(j>>8);
            uint value=mode==0u ? pos_u16(original+i*2u) : uint(as_type<ushort>(half(w-base)));
            pos_store16(entry+width,value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t==0u) previous+=ranks[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline uint pos_index(device const uchar *p, uint width) {
    return width==1u ? uint(p[0]):pos_u16(p);
}

kernel void wan_positional_decode(device const uint *offsets [[buffer(0)]],
    device const uchar *blob [[buffer(1)]], device const float *lut4 [[buffer(2)]],
    device const float *lut5 [[buffer(3)]], device float *output [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant uint &tile [[buffer(6)]],
    constant uint &capacity [[buffer(7)]], uint t [[thread_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]) {
    threadgroup uint pack_failed;
    if (t==0u) pack_failed=atomic_load_explicit(status+5,memory_order_relaxed)|
        atomic_load_explicit(status+6,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint begin=offsets[g],end=offsets[g+1u],width=tile==256u?1u:2u;
    bool invalid=pack_failed!=0u || begin>capacity || end>capacity || begin>end || end-begin<4u ||
        (begin&3u)!=0u || (end&3u)!=0u || (g==0u&&begin!=0u);
    uint precision=0u,mode=0u,n=0u,base_bytes=0u,unpadded=0u;
    // Do not form an out-of-range device pointer until offsets are admitted.
    if (!invalid) {
        precision=uint(blob[begin]);mode=uint(blob[begin+1u]);n=pos_u16(blob+begin+2u);
        invalid=(precision!=4u&&precision!=5u&&precision!=16u)||mode>1u||n>tile;
        if (!invalid) {
            base_bytes=precision==16u ? tile*2u:tile/256u*(8u+precision*32u);
            unpadded=4u+base_bytes+n*(width+2u);
            invalid=((unpadded+3u)&~3u)!=end-begin || (precision==16u&&n!=0u);
        }
    }
    if (invalid) {
        if (t==0u) atomic_fetch_add_explicit(status+7,1u,memory_order_relaxed);
        for (uint j=t;j<tile;j+=256u) output[g*tile+j]=as_type<float>(0x7fc00000u);
        return;
    }
    device const uchar *record=blob+begin, *base=record+4u, *entries=base+base_bytes;
    device const float *lut=precision==4u ? lut4:lut5;
    uint bad=0u;
    if (precision!=16u) {
        bad|=uint(!isfinite(lut[t]));
        if (t<tile/256u) bad|=uint(pos_bad_block(base+t*(8u+precision*32u)));
        for (uint k=t;k<n;k+=256u) {
            device const uchar *entry=entries+k*(width+2u);
            uint index=pos_index(entry,width);
            bad|=uint(index>=tile || (k>0u&&index<=pos_index(entry-(width+2u),width)));
            float value=mode==0u ? pos_bf16(entry+width) : float(as_type<half>(ushort(pos_u16(entry+width))));
            bad|=uint(!isfinite(value));
        }
    }
    for (uint j=unpadded+t;j<end-begin;j+=256u) bad|=uint(record[j]!=0u);
    float values[4];
    for (uint pass=0u;pass<tile/256u;++pass) {
        uint j=pass*256u+t;
        float v=precision==16u ? pos_bf16(base+j*2u):
            pos_decode(base+pass*(8u+precision*32u),lut,t,precision);
        if (precision!=16u) {
            uint lo=0u,hi=n;
            while (lo<hi) {uint mid=lo+(hi-lo)/2u;
                if (pos_index(entries+mid*(width+2u),width)<j) lo=mid+1u;else hi=mid;}
            if (lo<n&&pos_index(entries+lo*(width+2u),width)==j) {
                device const uchar *entry=entries+lo*(width+2u)+width;
                if (mode==0u) v=pos_bf16(entry);
                else v+=float(as_type<half>(ushort(pos_u16(entry))));
            }
        }
        bad|=uint(!isfinite(v));values[pass]=v;
    }
    threadgroup uint errors[256];
    errors[t]=bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
        if (t<s) errors[t]|=errors[t+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u&&errors[0]!=0u) atomic_fetch_add_explicit(status+7,1u,memory_order_relaxed);
    for (uint pass=0u;pass<tile/256u;++pass)
        output[g*tile+pass*256u+t]=errors[0]!=0u ? as_type<float>(0x7fc00000u):values[pass];
}
