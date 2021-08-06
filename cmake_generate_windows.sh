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

rm -rf $LOCAL_TEMP


cd build
#cmake -DCMAKE_CUDA_FLAGS="-arch=sm_52" -DGUI:BOOL=true -A x64 ../
cmake -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_52,code=[compute_52,lto_52]" -DGUI:BOOL=true -DLTO:BOOL=true -A x64 ../

