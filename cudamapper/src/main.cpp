#include <string>
#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    genomeworks::Index index_generator = genomeworks::Index();
    index_generator.generate_index(std::string(argv[1]));

    return  0;
}
