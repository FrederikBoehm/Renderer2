# MA Pathtracer
A physically based path tracer, using OptiX and CUDA, for surface and volume rendering. It is used in my master thesis with the title "Filtered Volumetric Representations of Surface Meshes for LOD Rendering".
## Windows
#### Setup
Required:
* CUDA: At least 11.0
* OptiX 7.4.0
* Visual Studio 2017
* CMake: At least 3.18

Run `cmake_generate_windows.sh -OPTIX_DIR <install_dir>` in source directory, where `<install_dir>` is the location of the OptiX installation. This loads all project dependencies  and runs the cmake project generation. Open the Visual Studio solution in the `build` directory and build the target `raytracer`. If the error ``identifier "__popcnt64" is undefined`` occurs during compilation, it might be necessary to enable the software implementation of the ``CountOn`` function in ``3rdparty/openvdb/nanovdb/nanovdb/NanoVDB.h``.

The directory `examples` contains a `config.json` file and a `scenedescription.json` file that can be used as a reference.
Options that can be set in `config.json`:
* General (Frame size, samples per pixel, gamma, ...)
* Filtering (Enabled, minimum voxel size, samples per voxel, optimization iterations, ...)
* Camera (Fov, pos, lookAt, up)
* Scene (Path to envmap, Path to scenedescription)

The `scenedescription.json` defines which model is located at which position.
All paths defined in these files are relative to the invocation path of the program.

#### Rendering
The `scenedescription.json` can be modified to render the models of interest.
When the compilation was successfull the target `raytracer` can be run. This will open the visualisation window and write `output.jpg` and `output.png` to `build/raytracer`. When running the program outside of Visual Studio it is important to pass the path of the `config.json` to the program.

#### Filtering
To enable filtering set the `Filtering->Active` property in the `config.json` to `true`.
Additionally the scenedescription defines the model that should be filtered using the mask `FILTERING`.
Note that only a single model with this mask is allowed at the same time.

## Linux
*Currently not supported*