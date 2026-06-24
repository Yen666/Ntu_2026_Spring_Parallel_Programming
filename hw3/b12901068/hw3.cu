#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.1415926535897932f
#define BLOCK_SIZE 16

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat3 mat3;

// Scene parameters structure - all in one for single cudaMemcpyToSymbol
struct SceneParams {
    float power;
    float md_iter;
    float ray_step;
    float shadow_step;
    float step_limiter;
    float ray_multiplier;
    float bailout;
    float eps;
    float FOV;
    float far_plane;
    vec3 camera_pos;
    vec3 target_pos;
    int AA;
};

// Constant memory for scene parameters (faster access)
__constant__ SceneParams d_scene;

// Device functions for vector operations
__device__ __forceinline__ float length(const vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ vec3 normalize(const vec3& v) {
    float len = length(v);
    return vec3(v.x / len, v.y / len, v.z / len);
}

__device__ __forceinline__ float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float clamp(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__device__ __forceinline__ vec3 clamp(const vec3& v, float minVal, float maxVal) {
    return vec3(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal));
}

// Fast integer power functions for power=8
__device__ __forceinline__ float powi8(float x) {
    float x2 = x * x;      // x^2
    float x4 = x2 * x2;    // x^4
    return x4 * x4;        // x^8
}

__device__ __forceinline__ float powi7(float x) {
    float x2 = x * x;      // x^2
    float x4 = x2 * x2;    // x^4
    return x4 * x2 * x;    // x^7
}

// Fast power for gloss=32 (x^32)
__device__ __forceinline__ float powi32(float x) {
    float x2 = x * x;      // x^2
    float x4 = x2 * x2;    // x^4
    float x8 = x4 * x4;    // x^8
    float x16 = x8 * x8;   // x^16
    return x16 * x16;      // x^32
}

// Mandelbulb distance estimator
// Mandelbulb distance estimator - only fast math optimization
__device__ float md(const vec3& p, float& trap) {
    vec3 v = p;
    float dr = 1.0f;
    float r = length(v);
    trap = r;

    for (int i = 0; i < d_scene.md_iter; ++i) {
        float theta = atan2f(v.y, v.x) * d_scene.power;
        float phi = asinf(v.z / r) * d_scene.power;
        dr = d_scene.power * powi7(r) * dr + 1.0f;
        
        float rp = powi8(r);
        float cosTheta, sinTheta;
        __sincosf(theta, &sinTheta, &cosTheta);  // Compute both at once
        float cosPhi, sinPhi;
        __sincosf(phi, &sinPhi, &cosPhi);        // Compute both at once
        
        v = p + vec3(rp * cosTheta * cosPhi, 
                     rp * cosPhi * sinTheta, 
                     -rp * sinPhi);

        trap = fminf(trap, r);
        r = length(v);
        if (r > d_scene.bailout) break;
    }
    return 0.5f * __logf(r) * r / dr;
}

// Scene mapping with rotation
__device__ float map(const vec3& p, float& trap, int& ID) {
    // Precomputed: cos(pi/2) = 0, sin(pi/2) = 1
    // Rotation matrix for 90 deg along X-axis: [1,0,0; 0,0,-1; 0,1,0]
    vec3 rp = vec3(p.x, -p.z, p.y);
    ID = 1;
    return md(rp, trap);
}

__device__ float map(const vec3& p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Color palette
__device__ vec3 pal(float t, const vec3& a, const vec3& b, const vec3& c, const vec3& d) {
    float cosVal = __cosf(2.0f * pi * (c.x * t + d.x));
    return vec3(a.x + b.x * cosVal,
                a.y + b.y * __cosf(2.0f * pi * (c.y * t + d.y)),
                a.z + b.z * __cosf(2.0f * pi * (c.z * t + d.z)));
}

// Soft shadow calculation
__device__ float softshadow(const vec3& ro, const vec3& rd, float k) {
    float res = 1.0f;
    float t = 0.001f;  // Start with small epsilon to avoid division by zero
    
    for (int i = 0; i < d_scene.shadow_step; ++i) {
        vec3 pos = vec3(ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t);
        float h = map(pos);
        float inv_t = 1.0f / t;  // Multiply instead of divide
        res = fminf(res, k * h * inv_t);
        if (res < 0.02f) return 0.02f;
        t += clamp(h, 0.001f, d_scene.step_limiter);
    }
    return clamp(res, 0.02f, 1.0f);
}

// Surface normal calculation
__device__ vec3 calcNor(const vec3& p) {
    vec2 e = vec2(d_scene.eps, 0.0f);
    return normalize(vec3(
        map(vec3(p.x + e.x, p.y, p.z)) - map(vec3(p.x - e.x, p.y, p.z)),
        map(vec3(p.x, p.y + e.x, p.z)) - map(vec3(p.x, p.y - e.x, p.z)),
        map(vec3(p.x, p.y, p.z + e.x)) - map(vec3(p.x, p.y, p.z - e.x))
    ));
}

// Ray tracing
__device__ float trace(const vec3& ro, const vec3& rd, float& trap, int& ID) {
    float t = 0.0f;
    float len = 0.0f;

    for (int i = 0; i < d_scene.ray_step; ++i) {
        vec3 pos = vec3(ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t);
        len = map(pos, trap, ID);
        if (fabsf(len) < d_scene.eps || t > d_scene.far_plane) break;
        t += len * d_scene.ray_multiplier;
    }
    return t < d_scene.far_plane ? t : -1.0f;
}

// Main rendering kernel - optimized with fast math and shared memory
__global__ void renderKernel(unsigned char* image, int width, int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate (height)
    
    if (i >= height || j >= width) return;
    
    // Shared memory to cache camera basis vectors (same for entire block)
    __shared__ vec3 s_cf, s_cs, s_cu, s_sd;
    __shared__ vec2 s_iResolution;
    
    // One thread computes shared values
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_iResolution = vec2(width, height);
        
        vec3 ta = d_scene.target_pos;
        vec3 ro = d_scene.camera_pos;
        s_cf = normalize(vec3(ta.x - ro.x, ta.y - ro.y, ta.z - ro.z));
        s_cs = normalize(cross(s_cf, vec3(0.0f, 1.0f, 0.0f)));
        s_cu = normalize(cross(s_cs, s_cf));
        s_sd = normalize(ro);  // light direction
    }
    __syncthreads();
    
    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;
    
    // Anti-aliasing loop
    for (int m = 0; m < d_scene.AA; ++m) {
        for (int n = 0; n < d_scene.AA; ++n) {
            vec2 p = vec2(j, i) + vec2(m, n) / (float)d_scene.AA;
            
            // Screen space to normalized coordinates - use shared memory
            vec2 uv = vec2((-s_iResolution.x + 2.0f * p.x) / s_iResolution.y,
                          -(-s_iResolution.y + 2.0f * p.y) / s_iResolution.y);
            
            // Camera setup - use precomputed shared memory values
            vec3 ro = d_scene.camera_pos;
            vec3 rd = normalize(vec3(uv.x * s_cs.x + uv.y * s_cu.x + d_scene.FOV * s_cf.x,
                                    uv.x * s_cs.y + uv.y * s_cu.y + d_scene.FOV * s_cf.y,
                                    uv.x * s_cs.z + uv.y * s_cu.z + d_scene.FOV * s_cf.z));
            
            // Ray marching
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);
            
            // Lighting and coloring
            vec3 col(0.0f);
            vec3 sc = vec3(1.0f, 0.9f, 0.717f);
            
            if (d < 0.0f) {
                // Sky color
                col = vec3(0.0f);
            } else {
                vec3 pos = vec3(ro.x + rd.x * d, ro.y + rd.y * d, ro.z + rd.z * d);
                vec3 nr = calcNor(pos);
                vec3 hal = normalize(vec3(s_sd.x - rd.x, s_sd.y - rd.y, s_sd.z - rd.z));
                
                // Color from orbit trap
                col = pal(trap - 0.4f, vec3(0.5f), vec3(0.5f), vec3(1.0f), vec3(0.0f, 0.1f, 0.2f));
                vec3 ambc = vec3(0.3f);
                float gloss = 32.0f;
                
                // Blinn-Phong lighting
                float amb = (0.7f + 0.3f * nr.y) * (0.2f + 0.8f * clamp(0.05f * __logf(trap), 0.0f, 1.0f));
                float sdw = softshadow(vec3(pos.x + 0.001f * nr.x, pos.y + 0.001f * nr.y, pos.z + 0.001f * nr.z), s_sd, 16.0f);
                float dif = clamp(dot(s_sd, nr), 0.0f, 1.0f) * sdw;
                float spe = powi32(clamp(dot(nr, hal), 0.0f, 1.0f)) * dif;  // Use powi32 instead of __powf
                
                vec3 lin(0.0f);
                lin = vec3(lin.x + ambc.x * (0.05f + 0.95f * amb),
                          lin.y + ambc.y * (0.05f + 0.95f * amb),
                          lin.z + ambc.z * (0.05f + 0.95f * amb));
                lin = vec3(lin.x + sc.x * dif * 0.8f,
                          lin.y + sc.y * dif * 0.8f,
                          lin.z + sc.z * dif * 0.8f);
                col = vec3(col.x * lin.x, col.y * lin.y, col.z * lin.z);
                
                // Fake SSS
                col = vec3(__powf(col.x, 0.7f), __powf(col.y, 0.9f), __powf(col.z, 1.0f));
                col = vec3(col.x + spe * 0.8f, col.y + spe * 0.8f, col.z + spe * 0.8f);
            }
            
            // Gamma correction
            col = clamp(vec3(__powf(col.x, 0.4545f), __powf(col.y, 0.4545f), __powf(col.z, 0.4545f)), 0.0f, 1.0f);
            fcol_r += col.x;
            fcol_g += col.y;
            fcol_b += col.z;
        }
    }
    
    // Average and convert to 0-255
    float aa_factor = (float)(d_scene.AA * d_scene.AA);
    fcol_r = (fcol_r / aa_factor) * 255.0f;
    fcol_g = (fcol_g / aa_factor) * 255.0f;
    fcol_b = (fcol_b / aa_factor) * 255.0f;
    
    // Write to image buffer (coalesced memory access)
    int idx = (i * width + j) * 4;
    image[idx + 0] = (unsigned char)fcol_r;
    image[idx + 1] = (unsigned char)fcol_g;
    image[idx + 2] = (unsigned char)fcol_b;
    image[idx + 3] = 255;
}

// Host function to write PNG - optimized for speed
void write_png(const char* filename, unsigned char* image, int width, int height) {
    // Ultra-fast settings: no compression at all
    LodePNGState state;
    lodepng_state_init(&state);
    
    // Disable ALL compression for maximum speed
    state.encoder.zlibsettings.btype = 0;  // No compression
    state.encoder.zlibsettings.use_lz77 = 0;  
    state.encoder.zlibsettings.windowsize = 32768;
    state.encoder.zlibsettings.minmatch = 3;
    state.encoder.zlibsettings.nicematch = 128;
    state.encoder.zlibsettings.lazymatching = 0;
    
    // Use PNG filter type 0 (None) for fastest encoding
    state.encoder.filter_palette_zero = 0;
    state.encoder.filter_strategy = LFS_ZERO;  // No filtering
    
    // Encode with custom settings
    unsigned char* png = nullptr;
    size_t pngsize = 0;
    unsigned error = lodepng_encode(&png, &pngsize, image, width, height, &state);
    
    if (!error) {
        lodepng_save_file(png, pngsize, filename);
    } else {
        printf("png error %u: %s\n", error, lodepng_error_text(error));
    }
    
    free(png);
    lodepng_state_cleanup(&state);
}

// Fast PPM (P6) output - no compression, much faster than PNG
void write_ppm(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: cannot open file %s\n", filename);
        return;
    }
    
    // PPM P6 header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write RGB data (skip alpha channel)
    for (int i = 0; i < width * height; ++i) {
        fwrite(&image[i * 4], 1, 3, fp);  // write R, G, B only
    }
    
    fclose(fp);
}

int main(int argc, char** argv) {
    assert(argc == 10);
    
    // Parse arguments
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int width = atoi(argv[7]);
    int height = atoi(argv[8]);
    
    // Scene parameters - pack into struct
    SceneParams scene;
    scene.power = 8.0f;
    scene.md_iter = 24.0f;
    scene.ray_step = 10000.0f;
    scene.shadow_step = 1500.0f;
    scene.step_limiter = 0.2f;
    scene.ray_multiplier = 0.1f;
    scene.bailout = 2.0f;
    scene.eps = 0.0005f;
    scene.FOV = 1.5f;
    scene.far_plane = 100.0f;
    scene.camera_pos = camera_pos;
    scene.target_pos = target_pos;
    scene.AA = 3;
    
    // Copy all parameters to constant memory in ONE call
    cudaMemcpyToSymbol(d_scene, &scene, sizeof(SceneParams));
    
    // Allocate device memory
    unsigned char* d_image;
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_image, image_size);
    
    // Allocate pinned host memory for faster D2H transfer
    unsigned char* h_image;
    cudaMallocHost(&h_image, image_size);  // Pinned memory instead of new[]
    
    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    renderKernel<<<gridSize, blockSize>>>(d_image, width, height);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        cudaFreeHost(h_image);
        return 1;
    }
    
    // Copy result back to host (using pinned memory for faster transfer)
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    
    // Check if output should be PPM (fast) or PNG (slow)
    const char* filename = argv[9];
    const char* ext = strrchr(filename, '.');
    
    if (ext && strcmp(ext, ".ppm") == 0) {
        write_ppm(filename, h_image, width, height);
    } else {
        write_png(filename, h_image, width, height);
    }
    
    // Cleanup
    cudaFree(d_image);
    cudaFreeHost(h_image);  // Free pinned memory
    
    return 0;
}