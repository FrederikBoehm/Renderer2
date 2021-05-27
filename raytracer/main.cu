#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "raytracer.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shapes/sphere.hpp"
#include "intersect/ray.hpp"
#include "shapes/plane.hpp"

__global__ void kernel(float* vector, float* matrix, float* output) {
  printf("%f", vector[0]);
  glm::vec4 glm_vector = glm::make_vec4(vector);
  glm::mat4 glm_matrix = glm::make_mat4(matrix);
  glm::vec4 glm_output = glm_matrix * glm_vector;
  memcpy(output, glm::value_ptr(glm_output), 4 * sizeof(float));

  rt::Sphere s(glm::vec3(1.0f, 2.0f, 3.0f), 5.0f);
  rt::Ray r(glm::vec3(0.0f, 2.0f, 3.0f), glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)));

  rt::SurfaceInteraction si = s.intersect(r);

  printf("Surface Intersection %i: x: %f, y: %f, z: %f", si.hit, si.pos.x, si.pos.y, si.pos.z);
}

void main() {
  using namespace rt;
  rt::Raytracer raytracer;

  glm::vec4 v(1.0f, 1.0f, 1.0f, 1.0f);
  glm::mat4 m(1.0f, 0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f, 0.0f,
              0.0f, 0.0f, 1.0f, 0.0f,
              4.0f, 0.5f, 2.0f, 1.0f);
  float* d_v;
  float* d_m;
  float* d_result;
  float h_result[4];
  float* vPtr = glm::value_ptr(m);

  cudaMalloc(&d_v, sizeof(float) * 4);
  cudaMalloc(&d_m, sizeof(float) * 16);
  cudaMalloc(&d_result, sizeof(float) * 4);

  cudaMemcpy(d_v, glm::value_ptr(v), sizeof(float) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, glm::value_ptr(m), sizeof(float) * 16, cudaMemcpyHostToDevice);

  kernel << <1, 1 >> > (d_v, d_m, d_result);

  cudaMemcpy(&h_result, d_result, sizeof(float) * 4, cudaMemcpyDeviceToHost);

  glm::vec4 glm_result = glm::make_vec4(h_result);

  cudaFree(d_v);
  cudaFree(d_m);
  cudaFree(d_result);

  glm::vec4 expected = m * v;

  
  rt::Sphere s(glm::vec3(1.0f, 2.0f, 3.0f), 5.0f);
  rt::Ray r(glm::vec3(0.0f, 2.0f, 3.0f), glm::normalize(glm::vec3(1.0f, -0.5f, 0.5f)));

  rt::SurfaceInteraction si = s.intersect(r);

  rt::Plane p(glm::vec3(1.0f, 2.0f, 3.0f), 10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
  rt::Ray r2(glm::vec3(0.0f, 0.0f, 0.0f), glm::normalize(glm::vec3(1.0f, 0.1f, 0.0f)));

  rt::SurfaceInteraction si2 = p.intersect(r2);



  while (true) {
    raytracer.renderFrame();
  }

}