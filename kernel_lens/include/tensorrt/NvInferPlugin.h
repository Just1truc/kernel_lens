#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"

namespace nvinfer1 {
namespace plugin {

TENSORRTAPI bool initLibNvInferPlugins(void* logger, const char* libNamespace);

} // namespace plugin
} // namespace nvinfer1

#endif // NV_INFER_PLUGIN_H
