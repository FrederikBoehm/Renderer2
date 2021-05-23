if not exist build mkdir build
cd build
cmake -DCMAKE_CUDA_FLAGS="-arch=sm_52" -A x64 ../