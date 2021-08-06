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

