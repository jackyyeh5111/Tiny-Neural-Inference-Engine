#include "onnx-ml.pb.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using TensorPtr = std::unique_ptr<onnx::TensorProto>;
using OperatorFunc = std::function<TensorPtr(const std::vector<const onnx::TensorProto*>&,
                                             const onnx::NodeProto&)>;
