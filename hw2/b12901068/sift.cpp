#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <cstring>
#include <chrono>
#include <mpi.h>

#include "sift.hpp"
#include "image.hpp"



ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                        Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    
    // MPI + OpenMP parallelization over octaves
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].resize(dog_pyramid.imgs_per_octave);
        
        // OpenMP parallel over scales
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            int scale_idx = j - 1;
            
            // MPI workload distribution
            if ((i * dog_pyramid.imgs_per_octave + scale_idx) % size != rank) {
                // Allocate empty image for proper indexing
                dog_pyramid.octaves[i][scale_idx] = Image(img_pyramid.octaves[i][j].width,
                                                          img_pyramid.octaves[i][j].height, 1);
                continue;
            }
            
            Image diff = img_pyramid.octaves[i][j];
            // Cache-friendly: use pointers for direct memory access
            const float* prev_data = img_pyramid.octaves[i][j-1].data;
            float* diff_data = diff.data;
            
            // Vectorizable loop with good spatial locality
            #pragma omp simd
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff_data[pix_idx] -= prev_data[pix_idx];
            }
            dog_pyramid.octaves[i][scale_idx] = diff;
        }
    }
    
    // MPI Bcast: collect DoG images from all processes
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        for (int j = 0; j < dog_pyramid.imgs_per_octave; j++) {
            // Determine which rank computed this scale
            int owner_rank = (i * dog_pyramid.imgs_per_octave + j) % size;
            int img_size = dog_pyramid.octaves[i][j].size;
            
            // Broadcast from owner to all processes
            MPI_Bcast(dog_pyramid.octaves[i][j].data, img_size, MPI_FLOAT, owner_rank, MPI_COMM_WORLD);
        }
    }
    
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    // Cache-optimized: use direct pointer access for better spatial locality
    const int width = img.width;
    const float* img_data = img.data;
    const float* prev_data = prev.data;
    const float* next_data = next.data;
    
    const int center = y * width + x;
    const float val = img_data[center];
    
    bool is_min = true, is_max = true;

    // Unrolled loop for better cache utilization and branch prediction
    // Check 3x3x3 neighborhood (27 points)
    for (int dy = -1; dy <= 1; dy++) {
        const int offset = (y + dy) * width;
        for (int dx = -1; dx <= 1; dx++) {
            const int idx = offset + x + dx;
            
            float neighbor = prev_data[idx];
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;
            
            neighbor = next_data[idx];
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;
            
            neighbor = img_data[idx];
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;
            
            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    std::vector<Keypoint> keypoints;
    
    // Distribute octaves across MPI processes
    #pragma omp parallel
    {
        std::vector<Keypoint> kps_local;
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < dog_pyramid.num_octaves; i++) {
            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                // MPI workload distribution: each process handles specific octaves
                if ((i * dog_pyramid.imgs_per_octave + j) % size != rank) {
                    continue;
                }
                
                const std::vector<Image>& octave = dog_pyramid.octaves[i];
                const Image& img = octave[j];
                
                // Cache-friendly: iterate y (rows) first, then x (columns)
                // This matches row-major memory layout for better spatial locality
                for (int y = 1; y < img.height-1; y++) {
                    for (int x = 1; x < img.width-1; x++) {
                        if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                            continue;
                        }
                        if (point_is_extremum(octave, j, x, y)) {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                          edge_thresh);
                            if (kp_is_valid) {
                                kps_local.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            keypoints.insert(keypoints.end(), kps_local.begin(), kps_local.end());
        }
    }
    
    
    // Gather keypoints from all MPI processes
    int local_count = keypoints.size();
    std::vector<int> counts(size);
    
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    int total_count = 0;
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        displs[i] = total_count;
        total_count += counts[i];
    }
    
    // Prepare data for MPI communication using float array
    std::vector<float> send_data(local_count * 136); // 4 ints + 4 floats + 128 descriptor
    
    // Cache-friendly: use pointer for sequential memory access
    float* send_ptr = send_data.data();
    for (int i = 0; i < local_count; i++) {
        const Keypoint& kp = keypoints[i];
        *send_ptr++ = kp.i;
        *send_ptr++ = kp.j;
        *send_ptr++ = kp.octave;
        *send_ptr++ = kp.scale;
        *send_ptr++ = kp.x;
        *send_ptr++ = kp.y;
        *send_ptr++ = kp.sigma;
        *send_ptr++ = kp.extremum_val;
        
        // Bulk copy descriptor (better cache locality)
        const uint8_t* desc = kp.descriptor.data();
        for (int j = 0; j < 128; j++) {
            *send_ptr++ = desc[j];
        }
    }
    
    std::vector<int> recv_counts(size), recv_displs(size);
    for (int i = 0; i < size; i++) {
        recv_counts[i] = counts[i] * 136;
        recv_displs[i] = displs[i] * 136;
    }
    
    std::vector<float> recv_data(total_count * 136);
    
    MPI_Allgatherv(send_data.data(), send_data.size(), MPI_FLOAT,
                   recv_data.data(), recv_counts.data(), recv_displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
    
    // Reconstruct keypoints (cache-friendly sequential access)
    std::vector<Keypoint> all_keypoints(total_count);
    const float* recv_ptr = recv_data.data();
    
    for (int i = 0; i < total_count; i++) {
        Keypoint& kp = all_keypoints[i];
        kp.i = (int)(*recv_ptr++);
        kp.j = (int)(*recv_ptr++);
        kp.octave = (int)(*recv_ptr++);
        kp.scale = (int)(*recv_ptr++);
        kp.x = *recv_ptr++;
        kp.y = *recv_ptr++;
        kp.sigma = *recv_ptr++;
        kp.extremum_val = *recv_ptr++;
        
        // Bulk copy descriptor (better cache locality)
        uint8_t* desc = kp.descriptor.data();
        for (int j = 0; j < 128; j++) {
            desc[j] = (uint8_t)(*recv_ptr++);
        }
    }
    
    
    // Print profiling info
    
    return all_keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    
    // MPI + OpenMP parallelization: distribute octaves across MPI processes
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].resize(pyramid.imgs_per_octave);
        
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        
        // OpenMP parallel over scales within each octave
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            // MPI workload distribution: each process handles specific scales
            if ((i * pyramid.imgs_per_octave + j) % size != rank) {
                // Still need to allocate empty image for proper indexing
                grad_pyramid.octaves[i][j] = Image(width, height, 2);
                continue;
            }
            
            Image grad(width, height, 2);
            const Image& img = pyramid.octaves[i][j];
            
            // Direct array access instead of set_pixel - much faster!
            float* grad_x = grad.data;  // channel 0
            float* grad_y = grad.data + width * height;  // channel 1
            
            // Parallelize gradient computation within each image
            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = y * width + x;
                    
                    // Compute x gradient (gx)
                    grad_x[idx] = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5f;
                    
                    // Compute y gradient (gy)
                    grad_y[idx] = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5f;
                }
            }
            grad_pyramid.octaves[i][j] = grad;
        }
    }
    
    // MPI Allgather: collect gradient images from all processes
    for (int i = 0; i < pyramid.num_octaves; i++) {
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        int img_size = width * height * 2; // 2 channels
        
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            // Determine which rank computed this scale
            int owner_rank = (i * pyramid.imgs_per_octave + j) % size;
            
            // Broadcast from owner to all processes
            MPI_Bcast(grad_pyramid.octaves[i][j].data, img_size, MPI_FLOAT, owner_rank, MPI_COMM_WORLD);
        }
    }
    
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        #pragma omp simd
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        #pragma omp simd
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    // Keep this serial - the patch is small and parallelization overhead isn't worth it
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

// Inline optimized version of update_histograms
// This function is called millions of times, so inlining and optimization are critical
inline void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                              float contrib, float theta_mn, float lambda_desc)
{
    // Precompute constants outside loops
    const float hist_scale = N_HIST * 0.5f / lambda_desc;
    const float bin_scale = N_ORI * 0.5f / M_PI;
    const float hist_step = 2.0f * lambda_desc / N_HIST;
    const float hist_center = (1.0f + N_HIST) * 0.5f;
    const float ori_step = 2.0f * M_PI / N_ORI;
    
    // Find range of bins that this sample contributes to
    // Only iterate over nearby bins (trilinear interpolation)
    const int i_min = std::max(1, (int)std::floor((x * N_HIST / (2.0f * lambda_desc)) + hist_center));
    const int i_max = std::min(N_HIST, (int)std::ceil((x * N_HIST / (2.0f * lambda_desc)) + hist_center));
    const int j_min = std::max(1, (int)std::floor((y * N_HIST / (2.0f * lambda_desc)) + hist_center));
    const int j_max = std::min(N_HIST, (int)std::ceil((y * N_HIST / (2.0f * lambda_desc)) + hist_center));
    
    for (int i = i_min; i <= i_max; i++) {
        const float x_i = (i - hist_center) * hist_step;
        const float dx = std::abs(x_i - x);
        if (dx > hist_step) continue;
        
        const float wx = 1.0f - hist_scale * dx;
        
        for (int j = j_min; j <= j_max; j++) {
            const float y_j = (j - hist_center) * hist_step;
            const float dy = std::abs(y_j - y);
            if (dy > hist_step) continue;
            
            const float hist_weight = wx * (1.0f - hist_scale * dy);
            
            // Orientation bins - unrolled for N_ORI=8
            #pragma omp simd
            for (int k = 1; k <= N_ORI; k++) {
                const float theta_k = ori_step * (k - 1);
                float theta_diff = theta_k - theta_mn;
                
                // Normalize to [0, 2*PI)
                if (theta_diff < 0) theta_diff += 2.0f * M_PI;
                if (theta_diff >= 2.0f * M_PI) theta_diff -= 2.0f * M_PI;
                
                const float abs_theta_diff = std::abs(theta_diff);
                if (abs_theta_diff < ori_step) {
                    const float bin_weight = 1.0f - bin_scale * abs_theta_diff;
                    hist[i-1][j-1][k-1] += hist_weight * bin_weight * contrib;
                }
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    // First pass: compute norm
    float norm = 0;
    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    
    // Second pass: clamp and compute norm2
    float threshold = 0.2f * norm;
    float norm2 = 0;
    #pragma omp simd reduction(+:norm2)
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], threshold);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    
    // Final pass: normalize and convert to uint8
    float scale = 512.0f / norm2;
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        float val = std::floor(scale * hist[i]);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    
    // Accumulate samples into histograms
    // Keep serial - update_histograms has complex dependencies
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<Keypoint> kps;
    
    
    // Distribute descriptor computation across MPI processes
    #pragma omp parallel
    {
        std::vector<Keypoint> kps_local;
        #pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < tmp_kps.size(); idx++) {
            // MPI workload distribution
            if (idx % size != rank) {
                continue;
            }
            
            Keypoint& kp_tmp = tmp_kps[idx];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                kps_local.push_back(kp);
            }
        }
        #pragma omp critical
        {
            kps.insert(kps.end(), kps_local.begin(), kps_local.end());
        }
    }
    
    // Gather descriptors from all MPI processes
    int local_count = kps.size();
    std::vector<int> counts(size);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    int total_count = 0;
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        displs[i] = total_count;
        total_count += counts[i];
    }
    
    // Serialize keypoints for MPI communication
    std::vector<float> send_buf(local_count * 136); // 4 int + 4 float + 128 descriptor
    for (int i = 0; i < local_count; i++) {
        int offset = i * 136;
        send_buf[offset + 0] = kps[i].i;
        send_buf[offset + 1] = kps[i].j;
        send_buf[offset + 2] = kps[i].octave;
        send_buf[offset + 3] = kps[i].scale;
        send_buf[offset + 4] = kps[i].x;
        send_buf[offset + 5] = kps[i].y;
        send_buf[offset + 6] = kps[i].sigma;
        send_buf[offset + 7] = kps[i].extremum_val;
        for (int j = 0; j < 128; j++) {
            send_buf[offset + 8 + j] = kps[i].descriptor[j];
        }
    }
    
    std::vector<int> recv_counts(size), recv_displs(size);
    for (int i = 0; i < size; i++) {
        recv_counts[i] = counts[i] * 136;
        recv_displs[i] = displs[i] * 136;
    }
    
    std::vector<float> recv_buf(total_count * 136);
    MPI_Allgatherv(send_buf.data(), send_buf.size(), MPI_FLOAT,
                   recv_buf.data(), recv_counts.data(), recv_displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
    std::vector<Keypoint> all_kps(total_count);
    // Deserialize keypoints (keep serial - fast enough and avoids potential OpenMP issues)
    for (int i = 0; i < total_count; i++) {
        int offset = i * 136;
        all_kps[i].i = (int)recv_buf[offset + 0];
        all_kps[i].j = (int)recv_buf[offset + 1];
        all_kps[i].octave = (int)recv_buf[offset + 2];
        all_kps[i].scale = (int)recv_buf[offset + 3];
        all_kps[i].x = recv_buf[offset + 4];
        all_kps[i].y = recv_buf[offset + 5];
        all_kps[i].sigma = recv_buf[offset + 6];
        all_kps[i].extremum_val = recv_buf[offset + 7];
        for (int j = 0; j < 128; j++) {
            all_kps[i].descriptor[j] = (uint8_t)recv_buf[offset + 8 + j];
        }
    }
    
    return all_kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    #pragma omp simd reduction(+:dist)
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < kps.size(); i++) {
        draw_point(res, kps[i].x, kps[i].y, 5);
    }
    return res;
}