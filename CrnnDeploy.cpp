/*
@author
date: 2020-03-17
Introduce:
    Deploy crnn model with libtorch.
*/

#include "CrnnDeploy.h"
#include <thread>
#include <sys/time.h>

//construtor
Crnn::Crnn(std::string& modelFile, std::string& keyFile){
    this->m_module = this->loadModule(modelFile);
    this->m_keys = this->readKeys(keyFile);
}


torch::Tensor Crnn::loadImg(std::string& imgFile, bool isbath){
	cv::Mat input = cv::imread(imgFile, 0);
	if(!input.data){
		printf("Error: not image data, imgFile input wrong!!");
	}
	int resize_h = int(input.cols * 32 / input.rows);
	cv::resize(input, input, cv::Size(resize_h, 32));
    torch::Tensor imgTensor;
    if(isbath){
        imgTensor = torch::from_blob(input.data, {32, resize_h, 1}, torch::kByte);
	    imgTensor = imgTensor.permute({2,0,1});
    }else
    {
        imgTensor = torch::from_blob(input.data, {1,32, resize_h, 1}, torch::kByte);
        imgTensor = imgTensor.permute({0,3,1,2});
    }
	imgTensor = imgTensor.toType(torch::kFloat);
	imgTensor = imgTensor.div(255);
	imgTensor = imgTensor.sub(0.5);
	imgTensor = imgTensor.div(0.5);
    return imgTensor;
}

void Crnn::infer(torch::Tensor& input){
    torch::Tensor output = this->m_module.forward({input}).toTensor();
    std::vector<int> predChars;
    int numImgs = output.sizes()[1];
    if(numImgs == 1){
        for(uint i=0; i<output.sizes()[0]; i++){
            auto maxRes = output[i].max(1, true);
            int maxIdx = std::get<1>(maxRes).item<float>();
            predChars.push_back(maxIdx);
        }
        // 字符转录处理
        std::string realChars="";
        for(uint i=0; i<predChars.size(); i++){
            if(predChars[i] != 0){
                if(!(i>0 && predChars[i-1]==predChars[i])){
                    realChars += this->m_keys[predChars[i]];
                }
            }
        }
        std::cout << realChars << std::endl;
    }else
    {
        std::vector<std::string> realCharLists;
        std::vector<std::vector<int>> predictCharLists;

        for (int i=0; i<output.sizes()[1]; i++){
            std::vector<int> temp;
            for(int j=0; j<output.sizes()[0]; j++){
                auto max_result = (output[j][i]).max(0, true);
                int max_index = std::get<1>(max_result).item<float>();//predict value
                temp.push_back(max_index);
            }
            predictCharLists.push_back(temp);
        }

        for(auto vec : predictCharLists){
            std::string text = "";
            for(uint i=0; i<vec.size(); i++){
                if(vec[i] != 0){
                    if(!(i>0 && vec[i-1]==vec[i])){
                        text += this->m_keys[vec[i]];
                    }
                }
            }
            realCharLists.push_back(text);
        }
        for(auto t : realCharLists){
            std::cout << t << std::endl;
        }
    }

}

std::vector<std::string> Crnn::readKeys(const std::string& keyFile){
    std::ifstream in(keyFile);
	std::ostringstream tmp;
	tmp << in.rdbuf();
	std::string keys = tmp.str();

    std::vector<std::string> words;
    words.push_back(" ");//函数过滤掉了第一个空格，这里加上
    int len = keys.length();
    int i = 0;
    
    while (i < len) {
      assert ((keys[i] & 0xF8) <= 0xF0);
      int next = 1;
      if ((keys[i] & 0x80) == 0x00) {
      } else if ((keys[i] & 0xE0) == 0xC0) {
        next = 2;
      } else if ((keys[i] & 0xF0) == 0xE0) {
        next = 3;
      } else if ((keys[i] & 0xF8) == 0xF0) {
        next = 4;
      }
      words.push_back(keys.substr(i, next));
      i += next;
    } 
    return words;
}

torch::jit::script::Module Crnn::loadModule(const std::string& modelFile){
    torch::jit::script::Module module;
    try{
         module = torch::jit::load(modelFile);
    }catch(const c10::Error& e){
        std::cerr << "error loadding the model !!!\n";
    }
    return module;
}


long getCurrentTime(void){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec/1000;
}

int main(int argc, const char* argv[]){

    if(argc<4){
        printf("Error use CrnnDeploy: loss input param !!! \n");
        return -1;
    }
    std::string modelFile = argv[1];
    std::string keyFile = argv[2];
    std::string imgFile = argv[3];

    long t1 = getCurrentTime();
    Crnn* crnn = new Crnn(modelFile,keyFile);
    torch::Tensor input = crnn->loadImg(imgFile);
    crnn->infer(input);
    delete crnn;
    long t2 = getCurrentTime();

    printf("ocr time : %ld ms \n", (t2-t1));
    return 0;
}
