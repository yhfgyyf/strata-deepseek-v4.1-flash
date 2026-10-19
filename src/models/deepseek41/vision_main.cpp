#include "strata/models/deepseek41/vision.hpp"
#include <fstream>
#include <iostream>

int main(int argc,char** argv){
    try{
        std::string model,patches,output,capture,integrity;int h=0,w=0;auto arithmetic=strata::models::deepseek41::Arithmetic::f32;
        for(int i=1;i<argc;++i){const std::string option=argv[i];auto next=[&]{if(i+1==argc)throw std::invalid_argument("vision option value");return std::string(argv[++i]);};
            if(option=="--mmproj")model=next();else if(option=="--patches")patches=next();else if(option=="--output")output=next();
            else if(option=="--integrity")integrity=next();
            else if(option=="--height")h=std::stoi(next());else if(option=="--width")w=std::stoi(next());else if(option=="--capture")capture=next();
            else if(option=="--arithmetic"){const auto a=next();if(a!="f32"&&a!="production")throw std::invalid_argument("vision arithmetic");arithmetic=a=="f32"?strata::models::deepseek41::Arithmetic::f32:strata::models::deepseek41::Arithmetic::production;}
            else throw std::invalid_argument("unknown vision option");}
        if(model.empty()||patches.empty()||output.empty()||h<1||w<1||int64_t(h)*w>9198)throw std::invalid_argument("vision input/grid/output");
        std::vector<float> pixels(int64_t(h)*w*588);std::ifstream in(patches,std::ios::binary);
        in.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*4);if(!in||in.peek()!=EOF)throw std::runtime_error("vision patch file size");
        auto values=strata::models::deepseek41::encode_vision(model,pixels,h,w,arithmetic,capture,integrity);
        std::ofstream out(output,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*4);if(!out)throw std::runtime_error("vision output file");
        std::cout<<"vision_tokens="<<values.size()/5120<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
