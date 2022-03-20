if [ ! -d "3rdparty" ]; then
	mkdir 3rdparty
fi

if [ ! -d "build" ]; then
	mkdir build
fi

LOCAL_TEMP="./build/temp"
mkdir $LOCAL_TEMP

(cd $LOCAL_TEMP && curl -o glad.zip -L --url https://github.com/Dav1dde/glad/archive/refs/tags/v0.1.34.zip && unzip -o "glad.zip" -d "glad")
mv "$LOCAL_TEMP/glad/glad-0.1.34" "3rdparty/glad"


(cd $LOCAL_TEMP && curl -o glfw.zip -L --url https://github.com/glfw/glfw/releases/download/3.3.4/glfw-3.3.4.zip && unzip -o "glfw.zip" -d "glfw")
mv "$LOCAL_TEMP/glfw/glfw-3.3.4" "3rdparty/glfw"


(cd $LOCAL_TEMP && curl -o glm.zip -L --url https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip && unzip -o "glm.zip" -d "glm")
mv "$LOCAL_TEMP/glm/glm" "3rdparty/glm"

(cd $LOCAL_TEMP && curl -o stb.zip -L --url https://github.com/nothings/stb/archive/80c8f6af0304588b9d780a41015472013b705194.zip && unzip -o "stb.zip" -d "stb")
mv "$LOCAL_TEMP/stb/stb-80c8f6af0304588b9d780a41015472013b705194" "3rdparty/stb"

(cd $LOCAL_TEMP && curl -o openvdb.zip -L --url https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v9.0.0.zip && unzip -o "openvdb.zip" -d "openvdb")
mv "$LOCAL_TEMP/openvdb/openvdb-9.0.0" "3rdparty/openvdb"

(cd $LOCAL_TEMP && curl -o vcpkg.zip -L --url https://github.com/microsoft/vcpkg/archive/refs/tags/2021.05.12.zip && unzip -o "vcpkg.zip" -d "vcpkg")
mv "$LOCAL_TEMP/vcpkg/vcpkg-2021.05.12" "3rdparty/vcpkg"

(cd $LOCAL_TEMP && curl -o libtiff.zip -L --url https://download.osgeo.org/libtiff/tiff-4.3.0.zip -k && unzip -o "libtiff.zip" -d "libtiff")
mv "$LOCAL_TEMP/libtiff/tiff-4.3.0" "3rdparty/libtiff"

(cd "3rdparty" && if [ ! -d "json" ]; then mkdir json; fi && cd json && curl -o json.hpp -L --url https://github.com/nlohmann/json/releases/download/v3.10.5/json.hpp)

(cd "3rdparty/vcpkg" && ./bootstrap-vcpkg.sh && \ 
	./vcpkg install zlib:x64-windows && \
	./vcpkg install blosc:x64-windows && \
	./vcpkg install tbb:x64-windows && \
	./vcpkg install boost-iostreams:x64-windows && \
	./vcpkg install boost-system:x64-windows && \
	./vcpkg install boost-any:x64-windows && \
	./vcpkg install boost-algorithm:x64-windows && \
	./vcpkg install boost-uuid:x64-windows && \
	./vcpkg install boost-interprocess:x64-windows)

rm -rf $LOCAL_TEMP


cd build
#cmake -DCMAKE_CUDA_FLAGS="-arch=sm_52" -DGUI:BOOL=true -A x64 ../
cmake -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_86" -DGUI:BOOL=true -DLTO:BOOL=true -A x64 ../

