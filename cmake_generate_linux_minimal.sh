if [ ! -d "3rdparty" ]; then
	mkdir 3rdparty
fi

if [ ! -d "build" ]; then
	mkdir build
fi

LOCAL_TEMP="./build/temp"
mkdir $LOCAL_TEMP


(cd $LOCAL_TEMP && curl -o glm.zip -L --url https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip && unzip -o "glm.zip" -d "glm")
mv "$LOCAL_TEMP/glm/glm" "3rdparty/glm"

(cd $LOCAL_TEMP && curl -o stb.zip -L --url https://github.com/nothings/stb/archive/80c8f6af0304588b9d780a41015472013b705194.zip && unzip -o "stb.zip" -d "stb")
mv "$LOCAL_TEMP/stb/stb-80c8f6af0304588b9d780a41015472013b705194" "3rdparty/stb"

(cd $LOCAL_TEMP && curl -o openvdb.zip -L --url https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v9.0.0.zip && unzip -o "openvdb.zip" -d "openvdb")
mv "$LOCAL_TEMP/openvdb/openvdb-9.0.0" "3rdparty/openvdb"

(cd $LOCAL_TEMP && curl -o vcpkg.zip -L --url https://github.com/microsoft/vcpkg/archive/refs/tags/2021.05.12.zip && unzip -o "vcpkg.zip" -d "vcpkg")
mv "$LOCAL_TEMP/vcpkg/vcpkg-2021.05.12" "3rdparty/vcpkg"

(cd "3rdparty" && if [ ! -d "json" ]; then mkdir json; fi && cd json && curl -o json.hpp -L --url https://github.com/nlohmann/json/releases/download/v3.10.5/json.hpp)

(cd "3rdparty/vcpkg" && ./bootstrap-vcpkg.sh && \ 
	./vcpkg install zlib:x64-linux && \
	./vcpkg install blosc:x64-linux && \
	./vcpkg install tbb:x64-linux && \
	./vcpkg install boost-iostreams:x64-linux && \
	./vcpkg install boost-system:x64-linux && \
	./vcpkg install boost-any:x64-linux && \
	./vcpkg install boost-algorithm:x64-linux && \
	./vcpkg install boost-uuid:x64-linux && \
	./vcpkg install boost-interprocess:x64-linux)

rm -rf $LOCAL_TEMP


cd build

case "$1" in
    -lto)
		cmake -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_75,code=[compute_75,lto_75]" -DCMAKE_BUILD_TYPE="Release" -DGUI:BOOL=false -DLTO:BOOL=true ../
        ;;
    *)
		cmake -DCMAKE_CUDA_FLAGS="-arch=sm_75" -DCMAKE_BUILD_TYPE="Release" -DGUI:BOOL=false -DLTO:BOOL=false ../
        ;;
esac

