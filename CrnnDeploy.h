
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cassert>
#include <vector>

#ifndef CRNN_H
#define CRNN_H

class Crnn{
    public:
        Crnn(std::string& modelFile, std::string& keyFile);
        torch::Tensor loadImg(std::string& imgFile, bool isbath=false);
        void infer(torch::Tensor& input);
    private:
        torch::jit::script::Module m_module;
        std::vector<std::string> m_keys;
        std::vector<std::string> readKeys(const std::string& keyFile);
        torch::jit::script::Module loadModule(const std::string& modelFile);
};

#endif//CRNN_H
