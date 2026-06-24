// Profile version of hw4.cu with detailed CUDA metrics
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include "sha256.h"

// ===== SHA256 helper macros for fully-unrolled mining kernel =====
// Use CUDA intrinsics for better performance on Ampere/Ada architectures
#if __CUDA_ARCH__ >= 700
#define ROTR32(x,n) __funnelshift_rc((x), (x), (n))
#else
#define ROTR32(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
#endif

#define CH32(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ32(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR32((x), 2) ^ ROTR32((x), 13) ^ ROTR32((x), 22))
#define EP1(x) (ROTR32((x), 6) ^ ROTR32((x), 11) ^ ROTR32((x), 25))
#define SIG0(x) (ROTR32((x), 7) ^ ROTR32((x), 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR32((x), 17) ^ ROTR32((x), 19) ^ ((x) >> 10))

extern __constant__ WORD k_device[64];

__device__ __forceinline__ WORD bswap32_dev(WORD v)
{
    #if __CUDA_ARCH__ >= 200
    return __byte_perm(v, 0, 0x0123);  // Ampere/Ada optimized
    #else
    return (v >> 24) | ((v & 0x00FF0000) >> 8) | ((v & 0x0000FF00) << 8) | (v << 24);
    #endif
}

__device__ __forceinline__ WORD load_be_word(const BYTE *ptr)
{
    return (WORD(ptr[0]) << 24) | (WORD(ptr[1]) << 16) | (WORD(ptr[2]) << 8) | WORD(ptr[3]);
}

#define SHA256_ROUND(a,b,c,d,e,f,g,h,w,k)                     \
    do {                                                     \
        WORD t1 = (h) + EP1(e) + CH32(e,f,g) + (k) + (w);    \
        WORD t2 = EP0(a) + MAJ32(a,b,c);                     \
        (h) = g;                                             \
        (g) = f;                                             \
        (f) = e;                                             \
        (e) = d + t1;                                        \
        (d) = c;                                             \
        (c) = b;                                             \
        (b) = a;                                             \
        (a) = t1 + t2;                                       \
    } while(0)

#define ROUND_0_15(i,a,b,c,d,e,f,g,h,w) \
    SHA256_ROUND(a,b,c,d,e,f,g,h, w[(i) & 15], k_device[i])

#define ROUND_16_63(i,a,b,c,d,e,f,g,h,w)                                \
    do {                                                                \
        w[(i) & 15] = w[((i) - 16) & 15] + SIG0(w[((i) - 15) & 15]) +   \
                      w[((i) - 7) & 15] + SIG1(w[((i) - 2) & 15]);     \
        SHA256_ROUND(a,b,c,d,e,f,g,h, w[(i) & 15], k_device[i]);        \
    } while(0)

#define DO8_ROUNDS_INIT(base,a,b,c,d,e,f,g,h,w) \
    ROUND_0_15(base+0,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+1,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+2,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+3,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+4,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+5,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+6,a,b,c,d,e,f,g,h,w);      \
    ROUND_0_15(base+7,a,b,c,d,e,f,g,h,w)

#define DO8_ROUNDS_EXP(base,a,b,c,d,e,f,g,h,w) \
    ROUND_16_63(base+0,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+1,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+2,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+3,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+4,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+5,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+6,a,b,c,d,e,f,g,h,w);     \
    ROUND_16_63(base+7,a,b,c,d,e,f,g,h,w)

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;

unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a': return 0x0a;
        case 'b': return 0x0b;
        case 'c': return 0x0c;
        case 'd': return 0x0d;
        case 'e': return 0x0e;
        case 'f': return 0x0f;
        case '0' ... '9': return c-'0';
    }
    return 0;
}

void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);
    size_t s = 0;
    size_t b = string_len/2-1;
    for(s, b; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

void getline(char *str, size_t len, FILE *fp)
{
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

__device__ __forceinline__ void double_sha256_megakernel(const WORD *midstate,
                                                         const BYTE *base_last12,
                                                         unsigned int nonce,
                                                         unsigned int hash_be[8])
{
    WORD w[16];
    w[0] = load_be_word(base_last12);
    w[1] = load_be_word(base_last12 + 4);
    w[2] = load_be_word(base_last12 + 8);
    w[3] = (WORD(nonce & 0xFF) << 24) | (WORD((nonce >> 8) & 0xFF) << 16) |
           (WORD((nonce >> 16) & 0xFF) << 8) | WORD((nonce >> 24) & 0xFF);
    w[4] = 0x80000000;
    w[5] = w[6] = w[7] = w[8] = w[9] = w[10] = w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 0x00000280;

    WORD a = midstate[0];
    WORD b = midstate[1];
    WORD c = midstate[2];
    WORD d = midstate[3];
    WORD e = midstate[4];
    WORD f = midstate[5];
    WORD g = midstate[6];
    WORD h = midstate[7];

    DO8_ROUNDS_INIT(0, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_INIT(8, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(16, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(24, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(32, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(40, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(48, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(56, a,b,c,d,e,f,g,h,w);

    WORD ihash[8];
    ihash[0] = midstate[0] + a;
    ihash[1] = midstate[1] + b;
    ihash[2] = midstate[2] + c;
    ihash[3] = midstate[3] + d;
    ihash[4] = midstate[4] + e;
    ihash[5] = midstate[5] + f;
    ihash[6] = midstate[6] + g;
    ihash[7] = midstate[7] + h;

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    w[0] = ihash[0]; w[1] = ihash[1]; w[2] = ihash[2]; w[3] = ihash[3];
    w[4] = ihash[4]; w[5] = ihash[5]; w[6] = ihash[6]; w[7] = ihash[7];
    w[8] = 0x80000000;
    w[9] = w[10] = w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 0x00000100;

    DO8_ROUNDS_INIT(0, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_INIT(8, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(16, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(24, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(32, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(40, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(48, a,b,c,d,e,f,g,h,w);
    DO8_ROUNDS_EXP(56, a,b,c,d,e,f,g,h,w);

    WORD final0 = 0x6a09e667 + a;
    WORD final1 = 0xbb67ae85 + b;
    WORD final2 = 0x3c6ef372 + c;
    WORD final3 = 0xa54ff53a + d;
    WORD final4 = 0x510e527f + e;
    WORD final5 = 0x9b05688c + f;
    WORD final6 = 0x1f83d9ab + g;
    WORD final7 = 0x5be0cd19 + h;

    hash_be[0] = bswap32_dev(final0);
    hash_be[1] = bswap32_dev(final1);
    hash_be[2] = bswap32_dev(final2);
    hash_be[3] = bswap32_dev(final3);
    hash_be[4] = bswap32_dev(final4);
    hash_be[5] = bswap32_dev(final5);
    hash_be[6] = bswap32_dev(final6);
    hash_be[7] = bswap32_dev(final7);
}

__global__ void mining_kernel(const WORD *midstate, unsigned char *target, 
                              unsigned int forward_start, unsigned int backward_end,
                              unsigned int *found_nonce, int *found_flag,
                              unsigned int nonces_per_thread, unsigned int total_threads,
                              const BYTE *base_last12)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned int s_target[8];
    __shared__ WORD s_midstate[8];
    __shared__ BYTE s_base_last12[12];
    __shared__ int s_found;
    
    // Thread 0 loads global found_flag into shared memory (減少 global memory access)
    if(threadIdx.x == 0) {
        s_found = *found_flag;
    }
    if(threadIdx.x < 8) {
        s_target[threadIdx.x] = ((unsigned int*)target)[threadIdx.x];
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
    }
    if(threadIdx.x < 12) {
        s_base_last12[threadIdx.x] = base_last12[threadIdx.x];
    }
    __syncthreads();
    
    // Early exit if already found
    if(s_found) return;
    
    // 單向搜尋：從 forward_start 往上找
    unsigned long long base_nonce_64 = (unsigned long long)forward_start + (unsigned long long)idx * nonces_per_thread;
    
    for(unsigned int offset = 0; offset < nonces_per_thread; ++offset)
    {
        // Check shared flag instead of global (避免 400-600 cycle latency)
        if(s_found) return;
        
        // 單向搜尋：只往上加
        unsigned long long nonce_64 = base_nonce_64 + offset;
        if(nonce_64 > 0xFFFFFFFF) return;  // 超出 32-bit 範圍
        
        unsigned int nonce = (unsigned int)nonce_64;
        
        unsigned int hash32[8];
        double_sha256_megakernel(s_midstate, s_base_last12, nonce, hash32);
        
        bool is_valid = true;
        for(int i=7; i>=0; --i) {
            if(hash32[i] > s_target[i]) { 
                is_valid = false;
                break;
            }
            else if(hash32[i] < s_target[i]) { 
                break;
            }
        }
        
        if(is_valid)
        {
            // 找到解：先嘗試寫入 global flag
            if(atomicCAS(found_flag, 0, 1) == 0)
            {
                *found_nonce = nonce;
            }
            // 更新 shared flag 通知同 block 的其他 threads
            s_found = 1;
            __syncthreads();
            return;
        }
    }
}

void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count;
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;

    while(total_count > 1)
    {
        int i, j;
        if(total_count % 2 == 1)
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            double_sha256((SHA256*)list[j], list[i], 64);
        }
        total_count = j;
    }

    memcpy(root, list[0], 32);
    delete[] raw_list;
    delete[] list;
}

struct ProfilingStats {
    float kernel_time_ms;
    float memcpy_h2d_ms;
    float memcpy_d2h_ms;
    unsigned long long nonces_tested;
    int kernel_launches;
};

void solve_with_profile(FILE *fin, FILE *fout, ProfilingStats &stats)
{
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;
    
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    unsigned char first64[64];
    memcpy(first64, &block.version, 4);
    memcpy(first64 + 4, block.prevhash, 32);
    memcpy(first64 + 36, block.merkle_root, 28);
    
    WORD midstate[8];
    sha256_midstate_cpu(midstate, first64);
    
    unsigned char base_last12[12];
    memcpy(base_last12, block.merkle_root + 28, 4);
    memcpy(base_last12 + 4, &block.ntime, 4);
    memcpy(base_last12 + 8, &block.nbits, 4);
    
    // Allocate device memory
    WORD *d_midstate;
    unsigned char *d_target;
    unsigned char *d_base_last12;
    unsigned int *d_found_nonce;
    int *d_found_flag;
    
    cudaMalloc(&d_midstate, 8 * sizeof(WORD));
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_base_last12, 12);
    cudaMalloc(&d_found_nonce, sizeof(unsigned int));
    cudaMalloc(&d_found_flag, sizeof(int));
    
    int found_flag = 0;
    unsigned int init_nonce = 0xFFFFFFFF;
    
    // Create CUDA events for timing
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);
    
    // Time H2D transfers
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &init_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_midstate, midstate, 8 * sizeof(WORD), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hex, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_last12, base_last12, 12, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    
    const unsigned int NONCES_PER_THREAD = 128;
    int threadsPerBlock = 256;
    int blocksPerGrid = 65536;
    unsigned int total_threads = threadsPerBlock * blocksPerGrid;
    unsigned int step_size = total_threads * NONCES_PER_THREAD;  // 單向搜尋，每次前進的步長
    
    SHA256 sha256_ctx;
    unsigned int found_nonce = 0;
    unsigned int forward_start = 0;
    
    stats.kernel_launches = 0;
    stats.nonces_tested = 0;
    float total_kernel_time = 0.0f;
    
    // 單向搜尋：從 0 到 0xFFFFFFFF
    while(forward_start <= 0xFFFFFFFF && !found_flag)
    {
        found_flag = 0;
        cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
        
        // Time kernel execution
        cudaEventRecord(start_kernel);
        // backward_end 不再使用，傳 0 即可
        mining_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_midstate, d_target, forward_start, 0, 
                                                           d_found_nonce, d_found_flag, NONCES_PER_THREAD, total_threads,
                                                           d_base_last12);
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);
        
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
        total_kernel_time += kernel_ms;
        stats.kernel_launches++;
        stats.nonces_tested += (unsigned long long)total_threads * NONCES_PER_THREAD;
        
        cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        if(found_flag)
        {
            cudaEventRecord(start_d2h);
            cudaMemcpy(&found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_d2h);
            cudaEventSynchronize(stop_d2h);
            
            block.nonce = found_nonce;
            double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
            break;
        }
        
        // 單向搜尋：只需要檢查是否超過範圍
        unsigned long long next_forward = (unsigned long long)forward_start + step_size;
        if(next_forward > 0xFFFFFFFF) break;
        forward_start = (unsigned int)next_forward;
    }
    
    // Synchronize and get timing results
    cudaEventSynchronize(stop_h2d);
    cudaEventSynchronize(stop_d2h);
    
    cudaEventElapsedTime(&stats.memcpy_h2d_ms, start_h2d, stop_h2d);
    cudaEventElapsedTime(&stats.memcpy_d2h_ms, start_d2h, stop_d2h);
    stats.kernel_time_ms = total_kernel_time;
    
    // Cleanup
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
    
    cudaFree(d_midstate);
    cudaFree(d_target);
    cudaFree(d_base_last12);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);

    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: profile_hw4 <in> <out>\n");
        return 1;
    }
    
    FILE *fin = fopen(argv[1], "r");
    if (fin == NULL) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", argv[1]);
        return 1;
    }
    FILE *fout = fopen(argv[2], "w");
    if (fout == NULL) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", argv[2]);
        fclose(fin);
        return 1;
    }

    int totalblock;
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);
    
    ProfilingStats total_stats = {0, 0, 0, 0, 0};
    
    for(int i=0;i<totalblock;++i)
    {
        ProfilingStats block_stats = {0, 0, 0, 0, 0};
        solve_with_profile(fin, fout, block_stats);
        
        total_stats.kernel_time_ms += block_stats.kernel_time_ms;
        total_stats.memcpy_h2d_ms += block_stats.memcpy_h2d_ms;
        total_stats.memcpy_d2h_ms += block_stats.memcpy_d2h_ms;
        total_stats.nonces_tested += block_stats.nonces_tested;
        total_stats.kernel_launches += block_stats.kernel_launches;
    }
    
    fclose(fin);
    fclose(fout);
    return 0;
}
