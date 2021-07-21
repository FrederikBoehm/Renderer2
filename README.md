# MAP Raytracer
An experimental Raytracer written in CUDA during the [Multicore Architectures and Programming (MAP) Seminar](https://www.cs12.tf.fau.de/lehre/lehrveranstaltungen/seminare/multi-core-architectures-and-programming/).
## Windows Setup
Required:
* CUDA: At least 11.0
* Visual Studio 2017
* CMake: At least 3.18

Run `cmake_generate_windows.sh` in source directory. This loads the dependencies [glad](https://github.com/Dav1dde/glad), [GLFW](https://github.com/glfw/glfw), [glm](https://github.com/g-truc/glm) and [stb](https://github.com/nothings/stb) and sets up the project with [Device Link Time Optimization](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/). Open Visual Studio solution in `build` directory and run target `ALL_BUILD`. Then the main target `raytracer` can be executed.

## Linux Setup
Required:
* CUDA: At least 11.0
* CMake: At least 3.18

Run `cmake_generate_linux_minimal.sh`, optionally with `-lto` flag for [Device Link Time Optimization](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/), in the source directory. This loads the dependencies [glm](https://github.com/g-truc/glm) and [stb](https://github.com/nothings/stb) and sets up the project. Change directory into `build` directory and run `cmake --build ./`. Execute the program with `./raytracer/raytracer`.