#include "strata/models/deepseek41/model.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: strata-deepseek41-inspect main.gguf [draft.gguf [mmproj.gguf]]\n"
                     "       strata-deepseek41-inspect main.gguf --operators-only\n";
        return 2;
    }
    try {
        if(argc==3 && std::string(argv[2])=="--operators-only"){
            const auto model=strata::models::deepseek41::Model::operator_weights_only(argv[1]);
            model->write_inventory(std::cout);
            return 0;
        }
        strata::models::deepseek41::Model model(argv[1], argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");
        model.write_inventory(std::cout);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
