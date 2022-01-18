# MT Pathtracer
A physically based path tracer, using OptiX and CUDA, for surface and volume rendering. It is used in my master thesis with the title "Filtered Volumetric Representations of Surface Meshes for LOD Rendering".
## Windows Setup
Required:
* CUDA: At least 11.0
* OptiX 7.4.0
* Visual Studio 2017
* CMake: At least 3.18

Run `cmake_generate_windows.sh` in source directory. This loads the dependencies [glad](https://github.com/Dav1dde/glad), [GLFW](https://github.com/glfw/glfw), [glm](https://github.com/g-truc/glm) and [stb](https://github.com/nothings/stb) and sets up the project with [Device Link Time Optimization](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/). Open Visual Studio solution in `build` directory and run target `ALL_BUILD`. Then the main target `raytracer` can be executed. This will open the visualisation window and write `output.jpg` and `output.png` to `build/raytracer`.

## Linux Setup
Required:
* CUDA: At least 11.0
* OptiX 7.4.0
* CMake: At least 3.18

Run `cmake_generate_linux_minimal.sh`, optionally with `-lto` flag for [Device Link Time Optimization](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/), in the source directory. This loads the dependencies [glm](https://github.com/g-truc/glm) and [stb](https://github.com/nothings/stb) and sets up the project. Change directory into `build` directory and run `cmake --build ./`. Execute the program with `./raytracer/raytracer`. This will write `output.jpg` and `output.png` to the `build` directory.