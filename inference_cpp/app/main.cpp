#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <memory>
#include <ctime>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

std::map<int, cv::Vec3b> colorMapping = {
    {0, cv::Vec3b(0, 0, 0)}, // Background (black)
    {1, cv::Vec3b(1, 1, 1)}, // Class 1 (red)
                             // Add more classes as needed.
};

cv::Mat decodeSegmentationToRGB(const cv::Mat &segmentationOutput)
{
  int rows = segmentationOutput.rows;
  int cols = segmentationOutput.cols;

  cv::Mat rgbImage(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int y = 0; y < rows; ++y)
  {
    for (int x = 0; x < cols; ++x)
    {
      int classLabel = (int)segmentationOutput.at<float>(y, x);
      if (colorMapping.find(classLabel) != colorMapping.end())
      {
        rgbImage.at<cv::Vec3b>(y, x) = colorMapping[classLabel];
      }
    }
  }
  return rgbImage;
}

int main()
{
  int w = 640;
  int h = 360;

  torch::jit::script::Module model;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load("/Users/guffrey/pytorch_cpp/car_seg/savedModel/model_traced.pt", torch::kCPU);
    model.eval();
    model.to(torch::kMPS);
    model = torch::jit::optimize_for_inference(model);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  cv::VideoCapture cap(1);
  if (!cap.isOpened())
  {
    std::cout << "Cannot open camera" << std::endl;
    return 1;
  }

  cv::Mat frame;
  while (true)
  {
    bool ret = cap.read(frame); // or cap >> frame;
    if (!ret)
    {
      std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
      break;
    }
    // auto start = std::chrono::system_clock::now();

    cv::resize(frame, frame, cv::Size{w, h});

    auto tensor_image = torch::from_blob(frame.data, {h, w, 3}, torch::kByte).permute({2, 0, 1});
    // std::cout << tensor_image.sizes() << std::endl;

    tensor_image = tensor_image.unsqueeze(0);
    tensor_image = tensor_image.to(torch::kF32).div(255.0);

    auto start = std::chrono::high_resolution_clock::now();
    tensor_image = tensor_image.to(torch::kMPS);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    auto output = model.forward(inputs).toTensor();
    output = output.to(torch::kCPU);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << std::endl;
    // output = output.toTensor();
    // output = output.argmax(1);
    // output = output.toType(torch::kFloat);
    // // output = output.to(torch::kCPU);
    // cv::Mat output_array(cv::Size{w, h}, CV_32F, output.data_ptr<float>());

    // cv::Mat rgbImage = decodeSegmentationToRGB(output_array);

    // auto end = std::chrono::system_clock::now();

    // std::chrono::duration<double> elapsed_seconds = end - start;

    // std::cout << elapsed_seconds.count() << std::endl;

    // frame = frame.mul(rgbImage);
    // cv::imshow("output", frame);

    if (cv::waitKey(1) == 'q')
    {
      break;
    }
  }
  return 0;
}