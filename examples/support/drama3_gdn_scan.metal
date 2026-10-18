#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Diagnostic only. All reductions retain ascending key order. Affine
// composition nevertheless changes F32 rounding relative to the recurrence.
inline float dgs_sigmoid(float x) {
    return x >= 0 ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}
inline float dgs_decay(float a, float alog, float dt) {
    float argument = a + dt;
    float sp = argument > 20 ? argument : log(1.0f + exp(argument));
    return exp(-exp(alog) * sp);
}

// Summary storage [block, value_head, key_column, basis_or_value_row].
// The first K rows start at identity and have v=0; the next V start at zero
// and use actual v. Each head has its own decay/beta, including shared k heads.
kernel void dgs_summary(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* a [[buffer(2)]], device const float* b [[buffer(3)]],
    device const float* alog [[buffer(4)]], device const float* dt [[buffer(5)]],
    device float* summary [[buffer(6)]], constant ulong* p [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    uint n=p[0], vh=p[1], vd=p[2], kh=p[3], kd=p[4], bs=p[5], blocks=p[6];
    uint width=kd+vd;
    if (id>=blocks*vh*width) return;
    uint row=id%width, h=(id/width)%vh, block=id/(width*vh), keyh=h/(vh/kh);
    float s[128];
    for(uint j=0;j<kd;j++) s[j]=row<kd && row==j ? 1.0f : 0.0f;
    for(uint t=block*bs;t<min(n,(block+1)*bs);t++) {
        float decay=dgs_decay(a[t*vh+h],alog[h],dt[h]);
        float beta=dgs_sigmoid(b[t*vh+h]), memory=0;
        uint kb=t*kh*kd+keyh*kd;
        for(uint j=0;j<kd;j++) { s[j]*=decay; memory+=s[j]*k[kb+j]; }
        float value=row<kd ? 0.0f : v[t*vh*vd+h*vd+row-kd];
        float delta=(value-memory)*beta;
        for(uint j=0;j<kd;j++) s[j]+=k[kb+j]*delta;
    }
    uint base=(block*vh+h)*kd*width;
    for(uint j=0;j<kd;j++) summary[base+j*width+row]=s[j];
}

// Hillis-Steele inclusive scan. Separate dispatches order each global level.
// Apply left (earlier) then right (later): (Al*Ar, Bl*Ar+Br).
kernel void dgs_compose(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant ulong* p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint vh=p[1], vd=p[2], kd=p[4], blocks=p[6], distance=p[7], width=kd+vd;
    uint per_head=kd*width, per_block=vh*per_head;
    if(id>=blocks*per_block) return;
    uint block=id/per_block;
    if(block<distance) { output[id]=input[id]; return; }
    uint row=id%width, column=(id/width)%kd, h=(id/per_head)%vh;
    uint left=((block-distance)*vh+h)*per_head, right=(block*vh+h)*per_head;
    float sum=0;
    for(uint j=0;j<kd;j++) sum+=input[left+j*width+row]*input[right+column*width+j];
    if(row>=kd) sum+=input[right+column*width+row];
    output[id]=sum;
}

// Incoming state of each replay block, physically [block,h,k,v]. Block zero
// copies S0 directly; later blocks apply the preceding inclusive prefix.
kernel void dgs_prefix_state(
    device const float* initial [[buffer(0)]], device const float* prefix [[buffer(1)]],
    device float* states [[buffer(2)]], constant ulong* p [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint vh=p[1], vd=p[2], kd=p[4], blocks=p[6], count=vh*kd*vd;
    if(id>=blocks*count) return;
    uint block=id/count, local=id%count;
    if(block==0) { states[id]=initial[local]; return; }
    uint value=local%vd, column=(local/vd)%kd, h=local/(kd*vd), width=kd+vd;
    uint base=((block-1)*vh+h)*kd*width;
    float sum=0;
    for(uint j=0;j<kd;j++) sum+=initial[(h*kd+j)*vd+value]*prefix[base+column*width+j];
    states[id]=sum+prefix[base+column*width+kd+value];
}

// Parallel across blocks and value lanes; sequential inside each block.
// Setting bs=n, blocks=1 gives the order-preserving single-launch control.
kernel void dgs_replay(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]], device const float* alog [[buffer(5)]],
    device const float* dt [[buffer(6)]], device const float* initial [[buffer(7)]],
    device float* output [[buffer(8)]], device float* final_state [[buffer(9)]],
    constant ulong* p [[buffer(10)]], uint id [[thread_position_in_grid]]) {
    uint n=p[0], vh=p[1], vd=p[2], kh=p[3], kd=p[4], bs=p[5], blocks=p[6];
    if(id>=blocks*vh*vd) return;
    uint value=id%vd, h=(id/vd)%vh, block=id/(vh*vd), keyh=h/(vh/kh);
    uint offset=h*kd*vd+value, state_count=vh*kd*vd;
    float s[128];
    for(uint j=0;j<kd;j++) s[j]=initial[block*state_count+offset+j*vd];
    for(uint t=block*bs;t<min(n,(block+1)*bs);t++) {
        float decay=dgs_decay(a[t*vh+h],alog[h],dt[h]);
        float beta=dgs_sigmoid(b[t*vh+h]), memory=0;
        uint kb=t*kh*kd+keyh*kd;
        for(uint j=0;j<kd;j++) { s[j]*=decay; memory+=s[j]*k[kb+j]; }
        float delta=(v[t*vh*vd+h*vd+value]-memory)*beta, sum=0;
        for(uint j=0;j<kd;j++) { s[j]+=k[kb+j]*delta; sum+=s[j]*q[kb+j]; }
        output[t*vh*vd+h*vd+value]=sum;
    }
    if(block+1==blocks)
        for(uint j=0;j<kd;j++) final_state[offset+j*vd]=s[j];
}
