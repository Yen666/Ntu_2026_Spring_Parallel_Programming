#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <thread>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

// GPU Kernels
__global__ void compute_accelerations_kernel(
    int n, int step, double dt, double G, double eps,
    const double* qx, const double* qy, const double* qz,
    const double* m, const int* is_device,
    double* ax, double* ay, double* az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double axi = 0, ayi = 0, azi = 0;
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        
        double mj = m[j];
        if (is_device[j]) {
            double t = step * dt;
            mj = mj + 0.5 * mj * fabs(sin(t / 6000.0));
        }
        
        double dx = qx[j] - qxi;
        double dy = qy[j] - qyi;
        double dz = qz[j] - qzi;
        double dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
        double dist3 = pow(dist_sq, 1.5);
        
        axi += G * mj * dx / dist3;
        ayi += G * mj * dy / dist3;
        azi += G * mj * dz / dist3;
    }
    
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

__global__ void update_velocities_kernel(
    int n, double dt,
    double* vx, double* vy, double* vz,
    const double* ax, const double* ay, const double* az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;
}

__global__ void update_positions_kernel(
    int n, double dt,
    double* qx, double* qy, double* qz,
    const double* vx, const double* vy, const double* vz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    qx[i] += vx[i] * dt;
    qy[i] += vy[i] * dt;
    qz[i] += vz[i] * dt;
}

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

#define BLOCK_SIZE 128
#define SMALL_WORKLOAD_THRESHOLD 100   // n < 200 使用 persistent kernel

// ====== 持久化 Kernel：在 GPU 內部完成多個 steps ======
// 適用於小 workload，減少 kernel launch overhead
__global__ void nbody_persistent_kernel_p1(
    int n, int total_steps, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, double* __restrict__ min_dist)
{
    // 使用 shared memory 儲存所有資料 - 對小 workload 完全可行
    extern __shared__ double shared_data[];
    double* s_qx = shared_data;
    double* s_qy = s_qx + n;
    double* s_qz = s_qy + n;
    double* s_vx = s_qz + n;
    double* s_vy = s_vx + n;
    double* s_vz = s_vy + n;
    double* s_m = s_vz + n;
    
    int tid = threadIdx.x;
    
    // 載入資料到 shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        s_qx[i] = qx[i];
        s_qy[i] = qy[i];
        s_qz[i] = qz[i];
        s_vx[i] = vx[i];
        s_vy[i] = vy[i];
        s_vz[i] = vz[i];
        s_m[i] = m[i];
    }
    __syncthreads();
    
    double local_min_dist = *min_dist;
    const double inv_6000 = 1.0 / 6000.0;  // 預計算常數
    const double eps_sq = eps * eps;        // 預計算 eps^2
    
    // 在 GPU 內部完成所有 steps
    for (int step = 1; step <= total_steps; step++) {
        double sin_val = sin(step * dt * inv_6000);
        double sin_abs = fabs(sin_val);
        double mass_factor = 1.0 + 0.5 * sin_abs;  // 預計算質量倍率
        
        // 每個 thread 處理一個或多個 bodies
        for (int i = tid; i < n; i += blockDim.x) {
            double qxi = s_qx[i], qyi = s_qy[i], qzi = s_qz[i];
            double ax = 0, ay = 0, az = 0;
            
            #pragma unroll 4
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                // 無分支計算有效質量
                double mj = s_m[j] * (is_device[j] ? mass_factor : 1.0);
                double dx = s_qx[j] - qxi;
                double dy = s_qy[j] - qyi;
                double dz = s_qz[j] - qzi;
                double dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double f = G * mj * inv_dist3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            
            s_vx[i] += ax * dt;
            s_vy[i] += ay * dt;
            s_vz[i] += az * dt;
        }
        __syncthreads();
        
        for (int i = tid; i < n; i += blockDim.x) {
            s_qx[i] += s_vx[i] * dt;
            s_qy[i] += s_vy[i] * dt;
            s_qz[i] += s_vz[i] * dt;
        }
        __syncthreads();
        
        // 計算最小距離 (只有 thread 0)
        if (tid == 0) {
            double dx = s_qx[planet] - s_qx[asteroid];
            double dy = s_qy[planet] - s_qy[asteroid];
            double dz = s_qz[planet] - s_qz[asteroid];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < local_min_dist) {
                local_min_dist = dist;
            }
        }
        __syncthreads();
    }
    
    // 寫回結果
    for (int i = tid; i < n; i += blockDim.x) {
        qx[i] = s_qx[i];
        qy[i] = s_qy[i];
        qz[i] = s_qz[i];
        vx[i] = s_vx[i];
        vy[i] = s_vy[i];
        vz[i] = s_vz[i];
    }
    if (tid == 0) {
        *min_dist = local_min_dist;
    }
}

// Problem 2 持久化 kernel
__global__ void nbody_persistent_kernel_p2(
    int n, int total_steps, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, double planet_radius_sq, int* __restrict__ hit_step)
{
    extern __shared__ double shared_data[];
    double* s_qx = shared_data;
    double* s_qy = s_qx + n;
    double* s_qz = s_qy + n;
    double* s_vx = s_qz + n;
    double* s_vy = s_vx + n;
    double* s_vz = s_vy + n;
    double* s_m = s_vz + n;
    
    int tid = threadIdx.x;
    
    for (int i = tid; i < n; i += blockDim.x) {
        s_qx[i] = qx[i]; s_qy[i] = qy[i]; s_qz[i] = qz[i];
        s_vx[i] = vx[i]; s_vy[i] = vy[i]; s_vz[i] = vz[i];
        s_m[i] = m[i];
    }
    __syncthreads();
    
    int local_hit = INT_MAX;
    const double inv_6000 = 1.0 / 6000.0;
    const double eps_sq = eps * eps;
    
    for (int step = 1; step <= total_steps; step++) {
        double sin_val = sin(step * dt * inv_6000);
        double sin_abs = fabs(sin_val);
        double mass_factor = 1.0 + 0.5 * sin_abs;
        
        for (int i = tid; i < n; i += blockDim.x) {
            double qxi = s_qx[i], qyi = s_qy[i], qzi = s_qz[i];
            double ax = 0, ay = 0, az = 0;
            #pragma unroll 4
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double mj = s_m[j] * (is_device[j] ? mass_factor : 1.0);
                double dx = s_qx[j] - qxi, dy = s_qy[j] - qyi, dz = s_qz[j] - qzi;
                double dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double f = G * mj * inv_dist3;
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            s_vx[i] += ax * dt; s_vy[i] += ay * dt; s_vz[i] += az * dt;
        }
        __syncthreads();
        
        for (int i = tid; i < n; i += blockDim.x) {
            s_qx[i] += s_vx[i] * dt; s_qy[i] += s_vy[i] * dt; s_qz[i] += s_vz[i] * dt;
        }
        __syncthreads();
        
        if (tid == 0 && local_hit == INT_MAX) {
            double dx = s_qx[planet] - s_qx[asteroid];
            double dy = s_qy[planet] - s_qy[asteroid];
            double dz = s_qz[planet] - s_qz[asteroid];
            if (dx*dx + dy*dy + dz*dz < planet_radius_sq) {
                local_hit = step;
            }
        }
        __syncthreads();
    }
    
    if (tid == 0 && local_hit != INT_MAX) {
        atomicMin(hit_step, local_hit);
    }
}

// Problem 3 持久化 kernel
__global__ void nbody_persistent_kernel_p3(
    int n, int total_steps, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, int device_id, 
    double missile_speed, double planet_radius_sq,
    int* __restrict__ destroy_step, int* __restrict__ hit_step)
{
    extern __shared__ double shared_data[];
    double* s_qx = shared_data;
    double* s_qy = s_qx + n;
    double* s_qz = s_qy + n;
    double* s_vx = s_qz + n;
    double* s_vy = s_vx + n;
    double* s_vz = s_vy + n;
    double* s_m = s_vz + n;
    
    int tid = threadIdx.x;
    
    for (int i = tid; i < n; i += blockDim.x) {
        s_qx[i] = qx[i]; s_qy[i] = qy[i]; s_qz[i] = qz[i];
        s_vx[i] = vx[i]; s_vy[i] = vy[i]; s_vz[i] = vz[i];
        s_m[i] = m[i];
    }
    __syncthreads();
    
    __shared__ int local_destroy;
    __shared__ int local_hit;
    if (tid == 0) {
        local_destroy = INT_MAX;
        local_hit = INT_MAX;
    }
    __syncthreads();
    
    const double inv_6000 = 1.0 / 6000.0;
    const double eps_sq = eps * eps;
    
    for (int step = 1; step <= total_steps; step++) {
        double sin_val = sin(step * dt * inv_6000);
        double sin_abs = fabs(sin_val);
        double mass_factor = 1.0 + 0.5 * sin_abs;
        
        for (int i = tid; i < n; i += blockDim.x) {
            double qxi = s_qx[i], qyi = s_qy[i], qzi = s_qz[i];
            double ax = 0, ay = 0, az = 0;
            #pragma unroll 4
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double mj = s_m[j] * (is_device[j] ? mass_factor : 1.0);
                double dx = s_qx[j] - qxi, dy = s_qy[j] - qyi, dz = s_qz[j] - qzi;
                double dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double f = G * mj * inv_dist3;
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            s_vx[i] += ax * dt; s_vy[i] += ay * dt; s_vz[i] += az * dt;
        }
        __syncthreads();
        
        for (int i = tid; i < n; i += blockDim.x) {
            s_qx[i] += s_vx[i] * dt; s_qy[i] += s_vy[i] * dt; s_qz[i] += s_vz[i] * dt;
        }
        __syncthreads();
        
        // Missile 和 collision 檢查 (只有 thread 0)
        if (tid == 0) {
            // Missile 檢查
            if (local_destroy == INT_MAX) {
                double dx = s_qx[device_id] - s_qx[planet];
                double dy = s_qy[device_id] - s_qy[planet];
                double dz = s_qz[device_id] - s_qz[planet];
                double dist = sqrt(dx*dx + dy*dy + dz*dz);
                double missile_reach = step * dt * missile_speed;
                if (missile_reach >= dist) {
                    local_destroy = step;
                    s_m[device_id] = 0;  // 裝置質量設為 0
                }
            }
            
            // Collision 檢查
            if (local_hit == INT_MAX) {
                double dx = s_qx[planet] - s_qx[asteroid];
                double dy = s_qy[planet] - s_qy[asteroid];
                double dz = s_qz[planet] - s_qz[asteroid];
                if (dx*dx + dy*dy + dz*dz < planet_radius_sq) {
                    local_hit = step;
                }
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *destroy_step = local_destroy;
        *hit_step = local_hit;
    }
}

// ====== 高平行度 kernel：多個 blocks 處理一個星體 ======
// 使用 2D grid: gridDim.x = n (星體數), gridDim.y = blocks_per_body
// 需要額外的 partial sums 陣列來存儲中間結果
__global__ void nbody_compute_accel_kernel(
    int n, int blocks_per_body, double sin_val, double G, double eps,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    double* __restrict__ partial_ax, double* __restrict__ partial_ay, double* __restrict__ partial_az)
{
    int i = blockIdx.x;  // 星體 index
    int block_id = blockIdx.y;  // 這個星體的第幾個 block
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    if (i >= n) return;
    
    __shared__ double s_ax[256];
    __shared__ double s_ay[256];
    __shared__ double s_az[256];
    
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    double ax = 0, ay = 0, az = 0;
    double sin_abs = fabs(sin_val);
    
    // 這個 block 負責的 j 範圍
    int total_threads = blocks_per_body * blockSize;
    int global_tid = block_id * blockSize + tid;
    
    for (int j = global_tid; j < n; j += total_threads) {
        if (j == i) continue;
        
        double mj = m[j];
        if (is_device[j]) {
            mj = mj + 0.5 * mj * sin_abs;
        }
        
        double dx = qx[j] - qxi;
        double dy = qy[j] - qyi;
        double dz = qz[j] - qzi;
        double dist_sq = dx * dx + dy * dy + dz * dz + eps * eps;
        
        double inv_dist = rsqrt(dist_sq);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;
        double f = G * mj * inv_dist3;
        
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    
    s_ax[tid] = ax;
    s_ay[tid] = ay;
    s_az[tid] = az;
    __syncthreads();
    
    // Block 內 reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ax[tid] += s_ax[tid + s];
            s_ay[tid] += s_ay[tid + s];
            s_az[tid] += s_az[tid + s];
        }
        __syncthreads();
    }
    
    // 儲存這個 block 的 partial sum
    if (tid == 0) {
        int idx = i * blocks_per_body + block_id;
        partial_ax[idx] = s_ax[0];
        partial_ay[idx] = s_ay[0];
        partial_az[idx] = s_az[0];
    }
}

// 第二階段：合併 partial sums 並更新位置
__global__ void nbody_update_kernel(
    int n, int blocks_per_body, double dt,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ partial_ax, const double* __restrict__ partial_ay, const double* __restrict__ partial_az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // 合併這個星體的所有 partial sums
    double ax = 0, ay = 0, az = 0;
    int base = i * blocks_per_body;
    for (int b = 0; b < blocks_per_body; b++) {
        ax += partial_ax[base + b];
        ay += partial_ay[base + b];
        az += partial_az[base + b];
    }
    
    // 更新速度和位置
    double nvx = vx[i] + ax * dt;
    double nvy = vy[i] + ay * dt;
    double nvz = vz[i] + az * dt;
    vx[i] = nvx;
    vy[i] = nvy;
    vz[i] = nvz;
    
    qx[i] += nvx * dt;
    qy[i] += nvy * dt;
    qz[i] += nvz * dt;
}

// ====== 簡單版 kernel (用於小 n 或不需要多 block) ======
// 一個 block 計算一個星體的加速度
__global__ void nbody_step_kernel_simple(
    int n, double sin_val, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ m, const int* __restrict__ is_device)
{
    int i = blockIdx.x;
    if (i >= n) return;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    __shared__ double s_ax[256];
    __shared__ double s_ay[256];
    __shared__ double s_az[256];
    
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    double ax = 0, ay = 0, az = 0;
    double sin_abs = fabs(sin_val);
    double mass_factor = 1.0 + 0.5 * sin_abs;
    double eps_sq = eps * eps;
    
    for (int j = tid; j < n; j += blockSize) {
        if (j == i) continue;
        
        // 無分支計算有效質量
        double mj = m[j] * (is_device[j] ? mass_factor : 1.0);
        
        double dx = qx[j] - qxi;
        double dy = qy[j] - qyi;
        double dz = qz[j] - qzi;
        double dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
        
        double inv_dist = rsqrt(dist_sq);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;
        double f = G * mj * inv_dist3;
        
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    
    s_ax[tid] = ax;
    s_ay[tid] = ay;
    s_az[tid] = az;
    __syncthreads();
    
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ax[tid] += s_ax[tid + s];
            s_ay[tid] += s_ay[tid + s];
            s_az[tid] += s_az[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        double nvx = vx[i] + s_ax[0] * dt;
        double nvy = vy[i] + s_ay[0] * dt;
        double nvz = vz[i] + s_az[0] * dt;
        vx[i] = nvx;
        vy[i] = nvy;
        vz[i] = nvz;
        
        qx[i] += nvx * dt;
        qy[i] += nvy * dt;
        qz[i] += nvz * dt;
    }
}

// run_step_gpu: 一個 block 一個星體
void run_step_gpu(int n, double sin_val,
    double* d_qx, double* d_qy, double* d_qz,
    double* d_vx, double* d_vy, double* d_vz,
    double* d_m, const int* d_is_device,
    hipStream_t stream = 0)
{
    nbody_step_kernel_simple<<<n, BLOCK_SIZE, 0, stream>>>(
        n, sin_val, param::dt, param::G, param::eps,
        d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_is_device);
}

// ========== Problem 1 合併版: nbody + min_dist ==========
__global__ void nbody_p1_kernel(
    int n, double sin_val, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, double* d_min_dist)
{
    int i = blockIdx.x;
    if (i >= n) return;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    __shared__ double s_ax[256];
    __shared__ double s_ay[256];
    __shared__ double s_az[256];
    
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    double ax = 0, ay = 0, az = 0;
    double sin_abs = fabs(sin_val);
    
    for (int j = tid; j < n; j += blockSize) {
        if (j == i) continue;
        double mj = m[j];
        if (is_device[j]) mj = mj + 0.5 * mj * sin_abs;
        double dx = qx[j] - qxi, dy = qy[j] - qyi, dz = qz[j] - qzi;
        double dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
        double inv_dist = rsqrt(dist_sq);
        double f = G * mj * inv_dist * inv_dist * inv_dist;
        ax += f * dx; ay += f * dy; az += f * dz;
    }
    
    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    
    for (int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) { s_ax[tid] += s_ax[tid+s]; s_ay[tid] += s_ay[tid+s]; s_az[tid] += s_az[tid+s]; }
        __syncthreads();
    }
    
    if (tid == 0) {
        double nvx = vx[i] + s_ax[0]*dt, nvy = vy[i] + s_ay[0]*dt, nvz = vz[i] + s_az[0]*dt;
        vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
        qx[i] += nvx*dt; qy[i] += nvy*dt; qz[i] += nvz*dt;
        
        // 只有處理 planet 的 block 計算 min_dist
        if (i == planet) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            double old = *d_min_dist;
            while (dist < old) {
                unsigned long long* addr = (unsigned long long*)d_min_dist;
                unsigned long long old_bits = __double_as_longlong(old);
                unsigned long long new_bits = __double_as_longlong(dist);
                unsigned long long result = atomicCAS(addr, old_bits, new_bits);
                if (result == old_bits) break;
                old = __longlong_as_double(result);
            }
        }
    }
}

// ========== Problem 2 合併版: nbody + collision ==========
__global__ void nbody_p2_kernel(
    int n, double sin_val, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, int step, double planet_radius_sq, int* d_hit_step)
{
    int i = blockIdx.x;
    if (i >= n) return;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    __shared__ double s_ax[256];
    __shared__ double s_ay[256];
    __shared__ double s_az[256];
    
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    double ax = 0, ay = 0, az = 0;
    double sin_abs = fabs(sin_val);
    
    for (int j = tid; j < n; j += blockSize) {
        if (j == i) continue;
        double mj = m[j];
        if (is_device[j]) mj = mj + 0.5 * mj * sin_abs;
        double dx = qx[j] - qxi, dy = qy[j] - qyi, dz = qz[j] - qzi;
        double dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
        double inv_dist = rsqrt(dist_sq);
        double f = G * mj * inv_dist * inv_dist * inv_dist;
        ax += f * dx; ay += f * dy; az += f * dz;
    }
    
    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    
    for (int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) { s_ax[tid] += s_ax[tid+s]; s_ay[tid] += s_ay[tid+s]; s_az[tid] += s_az[tid+s]; }
        __syncthreads();
    }
    
    if (tid == 0) {
        double nvx = vx[i] + s_ax[0]*dt, nvy = vy[i] + s_ay[0]*dt, nvz = vz[i] + s_az[0]*dt;
        vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
        qx[i] += nvx*dt; qy[i] += nvy*dt; qz[i] += nvz*dt;
        
        // 只有處理 planet 的 block 檢查碰撞
        if (i == planet) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx*dx + dy*dy + dz*dz < planet_radius_sq) {
                atomicMin(d_hit_step, step);
            }
        }
    }
}

// ========== Problem 3 合併版: nbody + missile + collision ==========
__global__ void nbody_p3_kernel(
    int n, double sin_val, double dt, double G, double eps,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    double* __restrict__ m, const int* __restrict__ is_device,
    int planet, int asteroid, int device_id, int step,
    double missile_speed, double planet_radius_sq,
    int* d_destroy_step, int* d_hit_step)
{
    int i = blockIdx.x;
    if (i >= n) return;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    __shared__ double s_ax[256];
    __shared__ double s_ay[256];
    __shared__ double s_az[256];
    
    double qxi = qx[i], qyi = qy[i], qzi = qz[i];
    double ax = 0, ay = 0, az = 0;
    double sin_abs = fabs(sin_val);
    
    for (int j = tid; j < n; j += blockSize) {
        if (j == i) continue;
        double mj = m[j];
        if (is_device[j]) mj = mj + 0.5 * mj * sin_abs;
        double dx = qx[j] - qxi, dy = qy[j] - qyi, dz = qz[j] - qzi;
        double dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
        double inv_dist = rsqrt(dist_sq);
        double f = G * mj * inv_dist * inv_dist * inv_dist;
        ax += f * dx; ay += f * dy; az += f * dz;
    }
    
    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    
    for (int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) { s_ax[tid] += s_ax[tid+s]; s_ay[tid] += s_ay[tid+s]; s_az[tid] += s_az[tid+s]; }
        __syncthreads();
    }
    
    if (tid == 0) {
        double nvx = vx[i] + s_ax[0]*dt, nvy = vy[i] + s_ay[0]*dt, nvz = vz[i] + s_az[0]*dt;
        vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
        qx[i] += nvx*dt; qy[i] += nvy*dt; qz[i] += nvz*dt;
        
        // 只有處理 planet 的 block 做檢查
        if (i == planet) {
            // 飛彈檢查
            double dx_pd = qx[device_id] - qx[planet];
            double dy_pd = qy[device_id] - qy[planet];
            double dz_pd = qz[device_id] - qz[planet];
            double dist_pd = sqrt(dx_pd*dx_pd + dy_pd*dy_pd + dz_pd*dz_pd);
            double missile_travel = step * missile_speed * dt;
            if (missile_travel >= dist_pd) {
                int old = atomicMin(d_destroy_step, step);
                if (step <= old) m[device_id] = 0.0;
            }
            
            // 碰撞檢查
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx*dx + dy*dy + dz*dz < planet_radius_sq) {
                atomicMin(d_hit_step, step);
            }
        }
    }
}

// GPU kernel: 计算距离并更新最小距离
__global__ void compute_min_dist_kernel(
    const double* qx, const double* qy, const double* qz,
    int planet, int asteroid, double* d_min_dist)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    
    // atomicMin for double - 使用CAS实现
    double old = *d_min_dist;
    while (dist < old) {
        unsigned long long* addr = (unsigned long long*)d_min_dist;
        unsigned long long old_bits = __double_as_longlong(old);
        unsigned long long new_bits = __double_as_longlong(dist);
        unsigned long long result = atomicCAS(addr, old_bits, new_bits);
        if (result == old_bits) break;
        old = __longlong_as_double(result);
    }
}

// GPU kernel: 检查碰撞
__global__ void check_collision_kernel(
    const double* qx, const double* qy, const double* qz,
    int planet, int asteroid, int step, double planet_radius_sq,
    int* d_hit_step)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    
    if (dist_sq < planet_radius_sq) {
        // 只记录第一次碰撞
        atomicMin(d_hit_step, step);
    }
}

// GPU kernel: Problem 3 - 檢查飛彈和碰撞
__global__ void check_missile_and_collision_kernel(
    const double* qx, const double* qy, const double* qz,
    double* m,  // 需要修改質量
    int planet, int asteroid, int device_id, int step,
    double missile_speed, double dt, double planet_radius_sq,
    int* d_destroy_step, int* d_hit_step)
{
    // 計算飛彈移動距離
    double missile_travel_distance = step * missile_speed * dt;
    
    // 計算行星到裝置的距離
    double dx_pd = qx[device_id] - qx[planet];
    double dy_pd = qy[device_id] - qy[planet];
    double dz_pd = qz[device_id] - qz[planet];
    double dist_planet_device = sqrt(dx_pd * dx_pd + dy_pd * dy_pd + dz_pd * dz_pd);
    
    // 檢查飛彈是否擊中裝置（只記錄第一次）
    if (missile_travel_distance >= dist_planet_device) {
        int old = atomicMin(d_destroy_step, step);
        // 如果這是第一次擊中，設置質量為0
        if (step <= old) {
            m[device_id] = 0.0;
        }
    }
    
    // 檢查小行星是否撞擊行星
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    
    if (dist_sq < planet_radius_sq) {
        atomicMin(d_hit_step, step);
    }
}

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    // 读取输入数据
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    
    // 找出所有 device 的索引
    std::vector<int> device_ids;
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            device_ids.push_back(i);
        }
    }

    // ====== GPU 0: Problem 1 + Problem 2 (使用兩個 stream) ======
    double min_dist = 1e308;  // 使用大數值代替 infinity()
    int hit_time_step = -2;
    
    // GPU 0 的資料 - Problem 1 (stream0) 和 Problem 2 (stream0b)
    hipStream_t stream0, stream0b;
    double *d_qx0, *d_qy0, *d_qz0, *d_vx0, *d_vy0, *d_vz0, *d_m0;
    int *d_is_device0;
    double *d_min_dist;
    
    double *d_qx0b, *d_qy0b, *d_qz0b, *d_vx0b, *d_vy0b, *d_vz0b, *d_m0b;
    int *d_is_device0b;
    int *d_hit_step;
    
    std::vector<double> m0 = m;  // Problem 1: device 質量為 0
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m0[i] = 0;
        }
    }
    std::vector<int> is_device0(n, 0);  // Problem 1: 不需要動態質量
    std::vector<int> is_device0b(n, 0);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            is_device0b[i] = 1;  // Problem 2: 需要動態質量
        }
    }
    
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamCreate(&stream0));
    HIP_CHECK(hipStreamCreate(&stream0b));
    
    // Problem 1 記憶體分配
    HIP_CHECK(hipMalloc(&d_qx0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_qy0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_qz0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vx0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_m0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_is_device0, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
    
    // Problem 2 記憶體分配
    HIP_CHECK(hipMalloc(&d_qx0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_qy0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_qz0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vx0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_m0b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_is_device0b, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
    
    // 複製資料 - Problem 1
    HIP_CHECK(hipMemcpyAsync(d_qx0, qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_qy0, qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_qz0, qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_vx0, vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_vy0, vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_vz0, vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_m0, m0.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0));
    HIP_CHECK(hipMemcpyAsync(d_is_device0, is_device0.data(), n * sizeof(int), hipMemcpyHostToDevice, stream0));
    double init_min_dist = 1e308;  // 使用大數值代替 infinity()
    HIP_CHECK(hipMemcpyAsync(d_min_dist, &init_min_dist, sizeof(double), hipMemcpyHostToDevice, stream0));
    
    // 複製資料 - Problem 2
    HIP_CHECK(hipMemcpyAsync(d_qx0b, qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_qy0b, qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_qz0b, qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_vx0b, vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_vy0b, vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_vz0b, vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_m0b, m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream0b));
    HIP_CHECK(hipMemcpyAsync(d_is_device0b, is_device0b.data(), n * sizeof(int), hipMemcpyHostToDevice, stream0b));
    int init_hit_step = INT_MAX;
    HIP_CHECK(hipMemcpyAsync(d_hit_step, &init_hit_step, sizeof(int), hipMemcpyHostToDevice, stream0b));
    
    double planet_radius_sq = param::planet_radius * param::planet_radius;
    
    // ====== GPU 1: Problem 3 - 使用多 streams 並行處理多個 devices ======
    int gravity_device_id = -1;
    double missile_cost = 1e308;  // 使用大數值代替 infinity()
    
    int num_devices = device_ids.size();
    
    // 為每個 device 創建獨立的資源
    std::vector<hipStream_t> streams1(num_devices);
    std::vector<double*> d_qx1_arr(num_devices), d_qy1_arr(num_devices), d_qz1_arr(num_devices);
    std::vector<double*> d_vx1_arr(num_devices), d_vy1_arr(num_devices), d_vz1_arr(num_devices);
    std::vector<double*> d_m1_arr(num_devices);
    std::vector<int*> d_is_device1_arr(num_devices);
    std::vector<int*> d_destroy_step_arr(num_devices), d_hit_step1_arr(num_devices);
    
    std::vector<int> is_device1(n, 0);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            is_device1[i] = 1;
        }
    }
    
    HIP_CHECK(hipSetDevice(1));
    
    // 為每個 device 分配 GPU 記憶體和 stream
    for (int d = 0; d < num_devices; d++) {
        HIP_CHECK(hipStreamCreate(&streams1[d]));
        HIP_CHECK(hipMalloc(&d_qx1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vx1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m1_arr[d], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_is_device1_arr[d], n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_destroy_step_arr[d], sizeof(int)));
        HIP_CHECK(hipMalloc(&d_hit_step1_arr[d], sizeof(int)));
    }
    
    // ====== 使用 std::thread 讓兩個 GPU 同時執行 ======
    
    // 根據 workload 大小選擇策略
    bool use_persistent = (n < SMALL_WORKLOAD_THRESHOLD);
    
    if (use_persistent) {
        // ====== 小 workload：使用持久化 kernel ======
        // Shared memory 大小：7 個 double 陣列 (qx, qy, qz, vx, vy, vz, m)
        size_t shared_mem_size = 7 * n * sizeof(double);
        
        // GPU 0 thread: Problem 1 + 2 (各用一個持久化 kernel)
        std::thread gpu0_thread([&]() {
            HIP_CHECK(hipSetDevice(0));
            
            // Problem 1：一次完成所有 steps
            nbody_persistent_kernel_p1<<<1, BLOCK_SIZE, shared_mem_size, stream0>>>(
                n, param::n_steps, param::dt, param::G, param::eps,
                d_qx0, d_qy0, d_qz0, d_vx0, d_vy0, d_vz0,
                d_m0, d_is_device0, planet, asteroid, d_min_dist);
            
            // Problem 2：一次完成所有 steps
            nbody_persistent_kernel_p2<<<1, BLOCK_SIZE, shared_mem_size, stream0b>>>(
                n, param::n_steps, param::dt, param::G, param::eps,
                d_qx0b, d_qy0b, d_qz0b, d_vx0b, d_vy0b, d_vz0b,
                d_m0b, d_is_device0b, planet, asteroid, planet_radius_sq, d_hit_step);
        });
        
        // GPU 1 thread: Problem 3 - 所有 devices 並行處理
        std::thread gpu1_thread([&]() {
            HIP_CHECK(hipSetDevice(1));
            
            // 為每個 device 啟動一個持久化 kernel
            for (int d = 0; d < num_devices; d++) {
                hipStream_t stream = streams1[d];
                int device_id = device_ids[d];
                
                // 複製初始資料
                HIP_CHECK(hipMemcpyAsync(d_qx1_arr[d], qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_qy1_arr[d], qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_qz1_arr[d], qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_vx1_arr[d], vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_vy1_arr[d], vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_vz1_arr[d], vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_m1_arr[d], m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_is_device1_arr[d], is_device1.data(), n * sizeof(int), hipMemcpyHostToDevice, stream));
                
                int init_destroy = INT_MAX;
                int init_hit1 = INT_MAX;
                HIP_CHECK(hipMemcpyAsync(d_destroy_step_arr[d], &init_destroy, sizeof(int), hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_hit_step1_arr[d], &init_hit1, sizeof(int), hipMemcpyHostToDevice, stream));
                
                // Problem 3：一次完成所有 steps
                nbody_persistent_kernel_p3<<<1, BLOCK_SIZE, shared_mem_size, stream>>>(
                    n, param::n_steps, param::dt, param::G, param::eps,
                    d_qx1_arr[d], d_qy1_arr[d], d_qz1_arr[d],
                    d_vx1_arr[d], d_vy1_arr[d], d_vz1_arr[d],
                    d_m1_arr[d], d_is_device1_arr[d],
                    planet, asteroid, device_id,
                    param::missile_speed, planet_radius_sq,
                    d_destroy_step_arr[d], d_hit_step1_arr[d]);
            }
            
            // 同步並獲取結果
            for (int d = 0; d < num_devices; d++) {
                HIP_CHECK(hipStreamSynchronize(streams1[d]));
                
                int destroy_step, hit_step1;
                HIP_CHECK(hipMemcpy(&destroy_step, d_destroy_step_arr[d], sizeof(int), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&hit_step1, d_hit_step1_arr[d], sizeof(int), hipMemcpyDeviceToHost));
                
                if (destroy_step != INT_MAX && hit_step1 == INT_MAX) {
                    double cost = param::get_missile_cost(destroy_step * param::dt);
                    if (cost < missile_cost) {
                        missile_cost = cost;
                        gravity_device_id = device_ids[d];
                    }
                }
            }
        });
        
        gpu0_thread.join();
        gpu1_thread.join();
        
    } else {
    // ====== 大 workload：使用分離式 kernel (nbody + 輕量檢查) ======
    // GPU 0 thread: Problem 1 + 2 (同時在兩個 streams 執行)
    std::thread gpu0_thread([&]() {
        HIP_CHECK(hipSetDevice(0));
        
        // Problem 1 和 Problem 2 同時在兩個 streams 執行
        for (int step = 1; step <= param::n_steps; step++) {
            double sin_val = sin(step * param::dt / 6000.0);
            
            // Problem 1: nbody 更新 + min_dist 檢查（輕量 kernel）
            nbody_step_kernel_simple<<<n, BLOCK_SIZE, 0, stream0>>>(
                n, sin_val, param::dt, param::G, param::eps,
                d_qx0, d_qy0, d_qz0, d_vx0, d_vy0, d_vz0, d_m0, d_is_device0);
            compute_min_dist_kernel<<<1, 1, 0, stream0>>>(
                d_qx0, d_qy0, d_qz0, planet, asteroid, d_min_dist);
            
            // Problem 2: nbody 更新 + collision 檢查（輕量 kernel）
            nbody_step_kernel_simple<<<n, BLOCK_SIZE, 0, stream0b>>>(
                n, sin_val, param::dt, param::G, param::eps,
                d_qx0b, d_qy0b, d_qz0b, d_vx0b, d_vy0b, d_vz0b, d_m0b, d_is_device0b);
            check_collision_kernel<<<1, 1, 0, stream0b>>>(
                d_qx0b, d_qy0b, d_qz0b, planet, asteroid, step, planet_radius_sq, d_hit_step);
        }
    });
    
    // GPU 1 thread: Problem 3 - 所有 devices 並行處理
    std::thread gpu1_thread([&]() {
        HIP_CHECK(hipSetDevice(1));
        
        // 同時為所有 devices 複製初始資料並啟動 kernels
        for (int d = 0; d < num_devices; d++) {
            hipStream_t stream = streams1[d];
            
            // 複製初始資料到 GPU 1
            HIP_CHECK(hipMemcpyAsync(d_qx1_arr[d], qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_qy1_arr[d], qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_qz1_arr[d], qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_vx1_arr[d], vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_vy1_arr[d], vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_vz1_arr[d], vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_m1_arr[d], m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_is_device1_arr[d], is_device1.data(), n * sizeof(int), hipMemcpyHostToDevice, stream));
            
            int init_destroy = INT_MAX;
            int init_hit1 = INT_MAX;
            HIP_CHECK(hipMemcpyAsync(d_destroy_step_arr[d], &init_destroy, sizeof(int), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_hit_step1_arr[d], &init_hit1, sizeof(int), hipMemcpyHostToDevice, stream));
        }
        
        // 主模擬迴圈 - 所有 devices 同時進行 (批次啟動 kernel)
        for (int step = 1; step <= param::n_steps; step++) {
            double sin_val = sin(step * param::dt / 6000.0);
            
            // 批次啟動所有 devices 的 nbody kernel
            for (int d = 0; d < num_devices; d++) {
                nbody_step_kernel_simple<<<n, BLOCK_SIZE, 0, streams1[d]>>>(
                    n, sin_val, param::dt, param::G, param::eps,
                    d_qx1_arr[d], d_qy1_arr[d], d_qz1_arr[d],
                    d_vx1_arr[d], d_vy1_arr[d], d_vz1_arr[d],
                    d_m1_arr[d], d_is_device1_arr[d]);
            }
            
            // 批次啟動所有 devices 的 check kernel
            for (int d = 0; d < num_devices; d++) {
                check_missile_and_collision_kernel<<<1, 1, 0, streams1[d]>>>(
                    d_qx1_arr[d], d_qy1_arr[d], d_qz1_arr[d], d_m1_arr[d],
                    planet, asteroid, device_ids[d], step,
                    param::missile_speed, param::dt, planet_radius_sq,
                    d_destroy_step_arr[d], d_hit_step1_arr[d]);
            }
        }
        
        // 同步所有 streams 並獲取結果
        for (int d = 0; d < num_devices; d++) {
            HIP_CHECK(hipStreamSynchronize(streams1[d]));
            
            int destroy_step, hit_step1;
            HIP_CHECK(hipMemcpy(&destroy_step, d_destroy_step_arr[d], sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&hit_step1, d_hit_step1_arr[d], sizeof(int), hipMemcpyDeviceToHost));
            
            // 如果裝置被摧毀且完全沒有發生撞擊，才算成功阻止
            if (destroy_step != INT_MAX && hit_step1 == INT_MAX) {
                double cost = param::get_missile_cost(destroy_step * param::dt);
                if (cost < missile_cost) {
                    missile_cost = cost;
                    gravity_device_id = device_ids[d];
                }
            }
        }
    });
    
    // 等待兩個 thread 完成
    gpu0_thread.join();
    gpu1_thread.join();
    
    }  // end of else (large workload)
    
    // ====== 獲取 GPU 0 的結果 ======
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(stream0));
    HIP_CHECK(hipStreamSynchronize(stream0b));
    HIP_CHECK(hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
    int gpu_hit_step;
    HIP_CHECK(hipMemcpy(&gpu_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
    hit_time_step = (gpu_hit_step == INT_MAX) ? -2 : gpu_hit_step;
    
    // ====== 清理 GPU 0 ======
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipFree(d_qx0));
    HIP_CHECK(hipFree(d_qy0));
    HIP_CHECK(hipFree(d_qz0));
    HIP_CHECK(hipFree(d_vx0));
    HIP_CHECK(hipFree(d_vy0));
    HIP_CHECK(hipFree(d_vz0));
    HIP_CHECK(hipFree(d_m0));
    HIP_CHECK(hipFree(d_is_device0));
    HIP_CHECK(hipFree(d_min_dist));
    HIP_CHECK(hipFree(d_qx0b));
    HIP_CHECK(hipFree(d_qy0b));
    HIP_CHECK(hipFree(d_qz0b));
    HIP_CHECK(hipFree(d_vx0b));
    HIP_CHECK(hipFree(d_vy0b));
    HIP_CHECK(hipFree(d_vz0b));
    HIP_CHECK(hipFree(d_m0b));
    HIP_CHECK(hipFree(d_is_device0b));
    HIP_CHECK(hipFree(d_hit_step));
    HIP_CHECK(hipStreamDestroy(stream0));
    HIP_CHECK(hipStreamDestroy(stream0b));
    
    // ====== 清理 GPU 1 ======
    HIP_CHECK(hipSetDevice(1));
    for (int d = 0; d < num_devices; d++) {
        HIP_CHECK(hipFree(d_qx1_arr[d]));
        HIP_CHECK(hipFree(d_qy1_arr[d]));
        HIP_CHECK(hipFree(d_qz1_arr[d]));
        HIP_CHECK(hipFree(d_vx1_arr[d]));
        HIP_CHECK(hipFree(d_vy1_arr[d]));
        HIP_CHECK(hipFree(d_vz1_arr[d]));
        HIP_CHECK(hipFree(d_m1_arr[d]));
        HIP_CHECK(hipFree(d_is_device1_arr[d]));
        HIP_CHECK(hipFree(d_destroy_step_arr[d]));
        HIP_CHECK(hipFree(d_hit_step1_arr[d]));
        HIP_CHECK(hipStreamDestroy(streams1[d]));
    }
    
    if (missile_cost >= 1e308) {  // 檢查是否仍為初始值
        missile_cost = 0;
        gravity_device_id = -1;
    }
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
