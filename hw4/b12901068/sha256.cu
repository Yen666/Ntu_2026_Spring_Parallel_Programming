
// This sha256 implementation is based on sha256 wiki page
// please refer to:
//     https://en.wikipedia.org/wiki/SHA-2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"

// circular shift - wiki:
//     https://en.wikipedia.org/wiki/Circular_shift
#define _rotl(v, s) ((v)<<(s) | (v)>>(32-(s)))
#define _rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))

#define _swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

// Use __constant__ memory for GPU, and static const for CPU
__constant__ WORD k_device[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static const WORD k_host[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Host and Device version - 優化 register 使用
__host__ __device__ void sha256_transform(SHA256 *ctx, const BYTE *msg)
{
	WORD i, j;
	
	// 使用滾動視窗減少 w 陣列大小 (從 64 降到 16)
	WORD w[16];
	
	// Initialize working variables to current hash value
	WORD a = ctx->h[0];
	WORD b = ctx->h[1];
	WORD c = ctx->h[2];
	WORD d = ctx->h[3];
	WORD e = ctx->h[4];
	WORD f = ctx->h[5];
	WORD g = ctx->h[6];
	WORD h = ctx->h[7];
	
	// Select k array based on whether we're on device or host
	#ifdef __CUDA_ARCH__
	const WORD *k = k_device;
	#else
	const WORD *k = k_host;
	#endif
	
	// Compress function main loop (展開成 16 的倍數以優化)
	// First 16 rounds: 直接從 msg 讀取
	for(i=0, j=0; i<16; ++i, j+=4)
	{
		int wi = i & 15;
		w[wi] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | (msg[j+3]);
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[wi];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}
	
	// Remaining 48 rounds: 使用滾動視窗計算 w
	for(i=16; i<64; ++i)
	{
		// 使用模運算重用 w 陣列空間
		WORD w_i_15 = w[(i-15) & 15];
		WORD w_i_2 = w[(i-2) & 15];
		WORD w_i_16 = w[(i-16) & 15];
		WORD w_i_7 = w[(i-7) & 15];
		
		WORD s0 = (_rotr(w_i_15, 7)) ^ (_rotr(w_i_15, 18)) ^ (w_i_15>>3);
		WORD s1 = (_rotr(w_i_2, 17)) ^ (_rotr(w_i_2, 19)) ^ (w_i_2>>10);
		
		// 更新當前 w 位置
		w[i & 15] = w_i_16 + s0 + w_i_7 + s1;
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[i & 15];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}
	
	// Add the compressed chunk to the current hash value
	ctx->h[0] += a;
	ctx->h[1] += b;
	ctx->h[2] += c;
	ctx->h[3] += d;
	ctx->h[4] += e;
	ctx->h[5] += f;
	ctx->h[6] += g;
	ctx->h[7] += h;
	
}

__host__ __device__ void sha256(SHA256 *ctx, const BYTE *msg, size_t len)
{
	// Initialize hash values:
	// (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
	ctx->h[0] = 0x6a09e667;
	ctx->h[1] = 0xbb67ae85;
	ctx->h[2] = 0x3c6ef372;
	ctx->h[3] = 0xa54ff53a;
	ctx->h[4] = 0x510e527f;
	ctx->h[5] = 0x9b05688c;
	ctx->h[6] = 0x1f83d9ab;
	ctx->h[7] = 0x5be0cd19;
	
	
	WORD i, j;
	size_t remain = len % 64;
	size_t total_len = len - remain;
	
	// Process the message in successive 512-bit chunks
	// For each chunk:
	for(i=0;i<total_len;i+=64)
	{
		sha256_transform(ctx, &msg[i]);
	}
	
	// Process remain data
	BYTE m[64] = {};
	for(i=total_len, j=0;i<len;++i, ++j)
	{
		m[j] = msg[i];
	}
	
	// Append a single '1' bit
	m[j++] = 0x80;  //1000 0000
	
	// Append K '0' bits, where k is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
	if(j > 56)
	{
		sha256_transform(ctx, m);
		memset(m, 0, sizeof(m));
		// printf("true\n"); // Commented out for device compatibility
	}
	
	// Append L as a 64-bit bug-endian integer, making the total post-processed length a multiple of 512 bits
	unsigned long long L = len * 8;  //bits
	m[63] = L;
	m[62] = L >> 8;
	m[61] = L >> 16;
	m[60] = L >> 24;
	m[59] = L >> 32;
	m[58] = L >> 40;
	m[57] = L >> 48;
	m[56] = L >> 56;
	sha256_transform(ctx, m);
	
	// Produce the final hash value (little-endian to big-endian)
	// Swap 1st & 4th, 2nd & 3rd byte for each word
	for(i=0;i<32;i+=4)
	{
        _swap(ctx->b[i], ctx->b[i+3]);
        _swap(ctx->b[i+1], ctx->b[i+2]);
	}
}

// Unit test
#ifdef __SHA256_UNITTEST__
	#define print_hash(x) printf("sha256 hash: "); for(int i=0;i<32;++i)printf("%02X", (x).b[i]);
	#define print_msg(x) printf("%s", ((x) ? "Pass":"Failed"))

int main(int argc, char **argv)
{
	SHA256 ctx;
	
	// ------------------ Stage 1: abc
	printf("------- Stage 1 : abc -------\n");
	BYTE abc[] = "abc";
	BYTE abcans[] = {0xBA, 0x78, 0x16, 0xBF, 0x8F, 0x01, 0xCF, 0xEA, 
					 0x41, 0x41, 0x40, 0xDE, 0x5D, 0xAE, 0x22, 0x23, 
					 0xB0, 0x03, 0x61, 0xA3, 0x96, 0x17, 0x7A, 0x9C, 
					 0xB4, 0x10, 0xFF, 0x61, 0xF2, 0x00, 0x15, 0xAD};
	size_t abclen = sizeof(abc) - 1;
	sha256(&ctx, abc, abclen);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(abcans, ctx.b, 32));
	printf("\n\n");
	
	// ------------------ Stage 2: len55
	printf("------ Stage 2 : len55 ------\n");
	BYTE len55[] = "1234567890123456789012345678901234567890123456789012345";
	BYTE len55ans[] = {0x03, 0xC3, 0xA7, 0x0E, 0x99, 0xED, 0x5E, 0xEC, 
					   0xCD, 0x80, 0xF7, 0x37, 0x71, 0xFC, 0xF1, 0xEC, 
					   0xE6, 0x43, 0xD9, 0x39, 0xD9, 0xEC, 0xC7, 0x6F, 
					   0x25, 0x54, 0x4B, 0x02, 0x33, 0xF7, 0x08, 0xE9};
	size_t len55len = sizeof(len55) - 1;
	sha256(&ctx, len55, len55len);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(len55ans, ctx.b, 32));
	printf("\n\n");
	
	// ------------------ Stage 3: len290
	printf("----- Stage 3 : len290 ------\n");
	BYTE len290[] = "ads;flkjas;dlkfjads;flkjads;flkafdlkjhfdalkjgadslfkjhadsjhfveroi"
					"uhwerpiuhwerptoiuywerptoiuywterypoihslgkjhdxzflgknbzsfdlkgjhsdfp"
					"gikjhwepgoiuhywertpiuywerptiuywrtoiuhwserlkjhsfdlgkjbsfd,nkmbxcv"
					".bkmnxflkjbnfdslgkjhsgpoiuhserpiuywerpituywetrpoiuhywerlkjbsfd,g"
					"nkbxsflkdjbsdflkjhsgfdluhsdgliuher";
	BYTE len290ans[] = {0xBD, 0xB5, 0xD4, 0xC1, 0xFB, 0x45, 0x1A, 0xD2, 
						0xFC, 0x8E, 0x62, 0x26, 0xF9, 0x5C, 0x6B, 0x58, 
						0x31, 0x53, 0x90, 0x1B, 0xE3, 0x74, 0xC2, 0x60, 
						0xC8, 0xA7, 0x46, 0x09, 0xC6, 0x89, 0x24, 0x60};
	size_t len290len = sizeof(len290) - 1;
	sha256(&ctx, len290, len290len);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(len290ans, ctx.b, 32));
	printf("\n\n");
	
	return 0;
}
#endif  //__SHA256_UNITTEST__

// ============================================================================
// 針對挖礦優化的特化版本：80 bytes block header 的 double SHA256
// ============================================================================

// 優化的 sha256_transform：只處理第一個 chunk（前 64 bytes）
__device__ void sha256_transform_80bytes_chunk1(SHA256 *ctx, const BYTE *msg)
{
	WORD i, j;
	WORD w[16];
	
	WORD a = ctx->h[0];
	WORD b = ctx->h[1];
	WORD c = ctx->h[2];
	WORD d = ctx->h[3];
	WORD e = ctx->h[4];
	WORD f = ctx->h[5];
	WORD g = ctx->h[6];
	WORD h = ctx->h[7];
	
	const WORD *k = k_device;
	
	// First 16 rounds
	for(i=0, j=0; i<16; ++i, j+=4)
	{
		int wi = i & 15;
		w[wi] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | (msg[j+3]);
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[wi];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	// Remaining 48 rounds
	#pragma unroll
	for(i=16; i<64; ++i)
	{
		WORD w_i_15 = w[(i-15) & 15];
		WORD w_i_2 = w[(i-2) & 15];
		WORD w_i_16 = w[(i-16) & 15];
		WORD w_i_7 = w[(i-7) & 15];
		
		WORD s0 = (_rotr(w_i_15, 7)) ^ (_rotr(w_i_15, 18)) ^ (w_i_15>>3);
		WORD s1 = (_rotr(w_i_2, 17)) ^ (_rotr(w_i_2, 19)) ^ (w_i_2>>10);
		w[i & 15] = w_i_16 + s0 + w_i_7 + s1;
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[i & 15];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
	ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

// 優化的 sha256_transform：處理第二個 chunk（最後 16 bytes + padding）
// 80 bytes 的情況：msg[0..15] 是最後 16 bytes，msg[16] = 0x80，msg[56..63] = length
__device__ void sha256_transform_80bytes_chunk2(SHA256 *ctx, const BYTE *last16)
{
	WORD i;
	WORD w[16];
	
	WORD a = ctx->h[0];
	WORD b = ctx->h[1];
	WORD c = ctx->h[2];
	WORD d = ctx->h[3];
	WORD e = ctx->h[4];
	WORD f = ctx->h[5];
	WORD g = ctx->h[6];
	WORD h = ctx->h[7];
	
	const WORD *k = k_device;
	
	// 手動構建 w[0..15]（固定 padding pattern for 80 bytes）
	// w[0..3]: 最後 16 bytes
	w[0] = (last16[0]<<24) | (last16[1]<<16) | (last16[2]<<8) | last16[3];
	w[1] = (last16[4]<<24) | (last16[5]<<16) | (last16[6]<<8) | last16[7];
	w[2] = (last16[8]<<24) | (last16[9]<<16) | (last16[10]<<8) | last16[11];
	w[3] = (last16[12]<<24) | (last16[13]<<16) | (last16[14]<<8) | last16[15];
	// w[4]: 0x80000000 (padding bit)
	w[4] = 0x80000000;
	// w[5..14]: 0x00000000
	w[5] = w[6] = w[7] = w[8] = w[9] = w[10] = w[11] = w[12] = w[13] = w[14] = 0;
	// w[15]: length = 80 * 8 = 640 bits = 0x00000280
	w[15] = 640;
	
	// First 16 rounds（直接使用預計算的 w）
	#pragma unroll
	for(i=0; i<16; ++i)
	{
		int wi = i & 15;
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[wi];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	// Remaining 48 rounds
	#pragma unroll
	for(i=16; i<64; ++i)
	{
		WORD w_i_15 = w[(i-15) & 15];
		WORD w_i_2 = w[(i-2) & 15];
		WORD w_i_16 = w[(i-16) & 15];
		WORD w_i_7 = w[(i-7) & 15];
		
		WORD s0 = (_rotr(w_i_15, 7)) ^ (_rotr(w_i_15, 18)) ^ (w_i_15>>3);
		WORD s1 = (_rotr(w_i_2, 17)) ^ (_rotr(w_i_2, 19)) ^ (w_i_2>>10);
		w[i & 15] = w_i_16 + s0 + w_i_7 + s1;
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[i & 15];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
	ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

// 第二次 SHA256：輸入固定 32 bytes
__device__ void sha256_32bytes(SHA256 *ctx, const SHA256 *input)
{
	// Initialize hash values
	ctx->h[0] = 0x6a09e667;
	ctx->h[1] = 0xbb67ae85;
	ctx->h[2] = 0x3c6ef372;
	ctx->h[3] = 0xa54ff53a;
	ctx->h[4] = 0x510e527f;
	ctx->h[5] = 0x9b05688c;
	ctx->h[6] = 0x1f83d9ab;
	ctx->h[7] = 0x5be0cd19;
	
	WORD i;
	WORD w[16];
	
	WORD a = ctx->h[0];
	WORD b = ctx->h[1];
	WORD c = ctx->h[2];
	WORD d = ctx->h[3];
	WORD e = ctx->h[4];
	WORD f = ctx->h[5];
	WORD g = ctx->h[6];
	WORD h = ctx->h[7];
	
	const WORD *k = k_device;
	
	#pragma unroll
	for(i=0; i<16; ++i)
	{
		int wi = i & 15;
		if(i < 8) {
			w[wi] = input->h[i];
		} else if(i == 8) {
			w[wi] = 0x80000000;
		} else if(i == 15) {
			w[wi] = 256; // 32 * 8
		} else {
			w[wi] = 0;
		}
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[wi];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	// Remaining 48 rounds
	#pragma unroll
	for(i=16; i<64; ++i)
	{
		WORD w_i_15 = w[(i-15) & 15];
		WORD w_i_2 = w[(i-2) & 15];
		WORD w_i_16 = w[(i-16) & 15];
		WORD w_i_7 = w[(i-7) & 15];
		
		WORD s0 = (_rotr(w_i_15, 7)) ^ (_rotr(w_i_15, 18)) ^ (w_i_15>>3);
		WORD s1 = (_rotr(w_i_2, 17)) ^ (_rotr(w_i_2, 19)) ^ (w_i_2>>10);
		w[i & 15] = w_i_16 + s0 + w_i_7 + s1;
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[i & 15];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
	ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
	
	// Convert to big-endian (byte swap)
	for(i=0; i<32; i+=4)
	{
		_swap(ctx->b[i], ctx->b[i+3]);
		_swap(ctx->b[i+1], ctx->b[i+2]);
	}
}

// 挖礦專用的 double SHA256：針對 80 bytes block header 優化
// 使用預計算的第一個 chunk 的中間狀態
__device__ void double_sha256_mining_precomputed(SHA256 *result, const WORD midstate[8], const BYTE *last16)
{
	SHA256 first_hash;
	
	// 使用預計算的 midstate（處理完前 64 bytes 後的狀態）
	first_hash.h[0] = midstate[0];
	first_hash.h[1] = midstate[1];
	first_hash.h[2] = midstate[2];
	first_hash.h[3] = midstate[3];
	first_hash.h[4] = midstate[4];
	first_hash.h[5] = midstate[5];
	first_hash.h[6] = midstate[6];
	first_hash.h[7] = midstate[7];
	
	// 只需處理最後 16 bytes + padding
	sha256_transform_80bytes_chunk2(&first_hash, last16);
	
	// Second SHA256 (input is 32 bytes)
	sha256_32bytes(result, &first_hash);
}

// 原版（不使用預計算）
__device__ void double_sha256_mining(SHA256 *result, const BYTE *block_header_80bytes)
{
	SHA256 first_hash;
	
	// Initialize hash values for first SHA256
	first_hash.h[0] = 0x6a09e667;
	first_hash.h[1] = 0xbb67ae85;
	first_hash.h[2] = 0x3c6ef372;
	first_hash.h[3] = 0xa54ff53a;
	first_hash.h[4] = 0x510e527f;
	first_hash.h[5] = 0x9b05688c;
	first_hash.h[6] = 0x1f83d9ab;
	first_hash.h[7] = 0x5be0cd19;
	
	// Process first 64 bytes
	sha256_transform_80bytes_chunk1(&first_hash, block_header_80bytes);
	
	// Process last 16 bytes + padding
	sha256_transform_80bytes_chunk2(&first_hash, block_header_80bytes + 64);
	
	// Second SHA256 (input is 32 bytes)
	sha256_32bytes(result, &first_hash);
}

// CPU 版本：計算前 64 bytes 的 midstate
void sha256_midstate_cpu(WORD midstate[8], const BYTE *first64)
{
	SHA256 ctx;
	ctx.h[0] = 0x6a09e667;
	ctx.h[1] = 0xbb67ae85;
	ctx.h[2] = 0x3c6ef372;
	ctx.h[3] = 0xa54ff53a;
	ctx.h[4] = 0x510e527f;
	ctx.h[5] = 0x9b05688c;
	ctx.h[6] = 0x1f83d9ab;
	ctx.h[7] = 0x5be0cd19;
	
	WORD i, j;
	WORD w[16];
	
	WORD a = ctx.h[0];
	WORD b = ctx.h[1];
	WORD c = ctx.h[2];
	WORD d = ctx.h[3];
	WORD e = ctx.h[4];
	WORD f = ctx.h[5];
	WORD g = ctx.h[6];
	WORD h = ctx.h[7];
	
	const WORD *k = k_host;
	
	// First 16 rounds
	for(i=0, j=0; i<16; ++i, j+=4)
	{
		int wi = i & 15;
		w[wi] = (first64[j]<<24) | (first64[j+1]<<16) | (first64[j+2]<<8) | first64[j+3];
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[wi];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	// Remaining 48 rounds
	for(i=16; i<64; ++i)
	{
		WORD w_i_15 = w[(i-15) & 15];
		WORD w_i_2 = w[(i-2) & 15];
		WORD w_i_16 = w[(i-16) & 15];
		WORD w_i_7 = w[(i-7) & 15];
		
		WORD s0 = (_rotr(w_i_15, 7)) ^ (_rotr(w_i_15, 18)) ^ (w_i_15>>3);
		WORD s1 = (_rotr(w_i_2, 17)) ^ (_rotr(w_i_2, 19)) ^ (w_i_2>>10);
		w[i & 15] = w_i_16 + s0 + w_i_7 + s1;
		
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD temp1 = h + S1 + ch + k[i] + w[i & 15];
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp2 = S0 + maj;
		
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	
	midstate[0] = ctx.h[0] + a;
	midstate[1] = ctx.h[1] + b;
	midstate[2] = ctx.h[2] + c;
	midstate[3] = ctx.h[3] + d;
	midstate[4] = ctx.h[4] + e;
	midstate[5] = ctx.h[5] + f;
	midstate[6] = ctx.h[6] + g;
	midstate[7] = ctx.h[7] + h;
}

#ifdef __cplusplus
}
#endif  //__cplusplus

#undef _rotl
#undef _rotr
