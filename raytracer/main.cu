#include <stdio.h>
#include "pipeline.hpp"

int main(int argc, char** argv) {
  
  if (argc < 2) {
    fprintf(stderr, "Error: Provide valid path to config file.\n");
    return 1;
  }
  
  CPipeline pipeline(argv[1]);
  pipeline.run();
  
  

  return 0;

}