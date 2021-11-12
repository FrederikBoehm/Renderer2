#include <stdio.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <fstream>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>
#include <vector>
#include <string>

int main() {
  const char* assetPath = "../../raytracer/assets/wdas_cloud/";
  std::vector<std::string> filenames = { "wdas_cloud_half",
                                         "wdas_cloud_quarter",
                                         "wdas_cloud_eighth",
                                         "wdas_cloud_sixteenth" };


  for (uint8_t i = 0; i < filenames.size(); ++i) {
    try {

      openvdb::initialize();

      std::ifstream ifile(assetPath + filenames[i] + ".vdb", std::ios_base::binary);
      auto grids = openvdb::io::Stream(ifile).getGrids();

      if (grids->size() == 0) {
        printf("No grid available\n");
        return 1;
      }
      else {
        auto srctype = grids->operator[](0)->type();
        auto handle = nanovdb::openToNanoVDB(grids->operator[](0));

        auto* dstGrid = handle.grid<float>();
        if (!dstGrid)
          throw std::runtime_error("GridHandle does not contain a grid with value type float");

        nanovdb::io::writeGrid(assetPath + filenames[i] + ".nvdb", handle);
      }
    }
    catch (const std::exception& e) {
      std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

  }


  return 0;
}