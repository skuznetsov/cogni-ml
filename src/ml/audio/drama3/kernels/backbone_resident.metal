#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline float drb_sigmoid(float v){return v>=0?1.0f/(1.0f+exp(-v)):exp(v)/(1.0f+exp(v));}
kernel void dr3_expert_sum(device const float* x [[buffer(0)]],
                         device const float* routing [[buffer(1)]],
                         device float* out [[buffer(2)]], constant ulong* p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float sum = 0;
    // Match the original zero-fill followed by ordered rank additions.
    for (uint rank = 0; rank < p[1]; rank++)
        sum += x[rank * p[0] + i] * routing[p[1] + rank];
    out[i] = sum;
}
kernel void dr3_route(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],constant ulong* p [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
 if(tid)return;uint n=uint(p[0]),k=uint(p[1]),ids[512];float scores[512],maximum=-INFINITY,total=0;for(uint i=0;i<n;i++)maximum=max(maximum,x[i]);
 for(uint i=0;i<n;i++){scores[i]=exp(x[i]-maximum);total+=scores[i];}for(uint i=0;i<n;i++)scores[i]/=total;
 float selected=0;for(uint r=0;r<k;r++){uint best=0;float value=-INFINITY;for(uint i=0;i<n;i++){bool used=false;for(uint j=0;j<r;j++)used|=ids[j]==i;if(!used&&scores[i]>value){best=i;value=scores[i];}}ids[r]=best;out[r]=float(best);selected+=scores[best];}
 for(uint r=0;r<k;r++)out[k+r]=scores[ids[r]]/(p[2]?selected:1.0f);
}
kernel void dr3_weighted_add(device float* out [[buffer(0)]],device const float* x [[buffer(1)]],device const float* weight [[buffer(2)]],constant ulong* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {if(i<p[0])out[i]+=x[i]*(p[1]?drb_sigmoid(weight[0]):weight[0]);}
kernel void dr3_split_q_gate(device const float* in [[buffer(0)]],device float* q [[buffer(1)]],device float* gate [[buffer(2)]],constant ulong* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {if(i<p[0]){uint h=i/uint(p[1]),d=i%uint(p[1]);q[i]=in[h*uint(p[1])*2+d];gate[i]=in[h*uint(p[1])*2+uint(p[1])+d];}}
kernel void dr3_full_attention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],device const float* gate [[buffer(3)]],device float* out [[buffer(4)]],constant ulong* p [[buffer(5)]],uint h [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
 threadgroup float scores[1024];threadgroup float denom;uint dim=uint(p[2]),len=uint(p[3]),kh=h/(uint(p[0])/uint(p[1])),stride=uint(p[1])*dim;
 for(uint t=lane;t<len;t+=64){float dot=0;for(uint j=0;j<dim;j++)dot+=q[h*dim+j]*k[t*stride+kh*dim+j];scores[t]=dot/sqrt(float(dim));}threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lane==0){float maximum=-INFINITY;for(uint t=0;t<len;t++)maximum=max(maximum,scores[t]);float sum=0;for(uint t=0;t<len;t++){scores[t]=exp(scores[t]-maximum);sum+=scores[t];}denom=sum;}threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint d=lane;d<dim;d+=64){float sum=0;for(uint t=0;t<len;t++)sum+=(scores[t]/denom)*v[t*stride+kh*dim+d];out[h*dim+d]=sum*drb_sigmoid(gate[h*dim+d]);}
}
kernel void dr3_delta_conv(device const float* history [[buffer(0)]],device const float* w [[buffer(1)]],device float* out [[buffer(2)]],constant ulong* p [[buffer(3)]],uint d [[thread_position_in_grid]]) {
 uint width=uint(p[0]),taps=uint(p[1]),pos=uint(p[2]);if(d>=width)return;uint len=min(pos+1,taps),first=pos+1-len;float sum=0;
 for(uint t=0;t<len;t++)sum+=history[((first+t)%taps)*width+d]*w[d*taps+taps-len+t];out[d]=sum*drb_sigmoid(sum);
}
kernel void dr3_delta_scale(device float* q [[buffer(0)]],device float* k [[buffer(1)]],constant ulong* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]){q[i]/=float(p[1]);k[i]/=sqrt(float(p[1]));}}
kernel void dr3_delta_step(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],device const float* a [[buffer(3)]],device const float* b [[buffer(4)]],device const float* alog [[buffer(5)]],device const float* dt [[buffer(6)]],device float* state [[buffer(7)]],device float* out [[buffer(8)]],constant ulong* p [[buffer(9)]],uint id [[thread_position_in_grid]]) {
 uint vh=uint(p[0]),vd=uint(p[1]),kh=uint(p[2]),kd=uint(p[3]);if(id>=vh*vd)return;uint h=id/vd,keyh=h/(vh/kh),offset=id*kd;
 float argument=a[h]+dt[h],sp=argument>20?argument:log(1.0f+exp(argument)),decay=exp(-exp(alog[h])*sp),beta=drb_sigmoid(b[h]);float memory=0;
 for(uint j=0;j<kd;j++){state[offset+j]*=decay;memory+=state[offset+j]*k[keyh*kd+j];}float delta=(v[id]-memory)*beta,sum=0;
 for(uint j=0;j<kd;j++){state[offset+j]+=k[keyh*kd+j]*delta;sum+=state[offset+j]*q[keyh*kd+j];}out[id]=sum;
}

// Value lanes are contiguous in the private recurrent state. Each lane keeps
// the original key-order reductions; only the physical state index changes.
kernel void dr3_delta_step_transposed(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]], device const float* alog [[buffer(5)]],
    device const float* dt [[buffer(6)]], device float* state [[buffer(7)]],
    device float* out [[buffer(8)]], constant ulong* p [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    uint vh = uint(p[0]), vd = uint(p[1]), kh = uint(p[2]), kd = uint(p[3]);
    if (id >= vh * vd) return;
    uint h = id / vd, keyh = h / (vh / kh), offset = h * kd * vd + id % vd;
    float argument = a[h] + dt[h];
    float sp = argument > 20 ? argument : log(1.0f + exp(argument));
    float decay = exp(-exp(alog[h]) * sp), beta = drb_sigmoid(b[h]);
    float memory = 0;
    for (uint j = 0; j < kd; j++) {
        state[offset + j * vd] *= decay;
        memory += state[offset + j * vd] * k[keyh * kd + j];
    }
    float delta = (v[id] - memory) * beta, sum = 0;
    for (uint j = 0; j < kd; j++) {
        state[offset + j * vd] += k[keyh * kd + j] * delta;
        sum += state[offset + j * vd] * q[keyh * kd + j];
    }
    out[id] = sum;
}
