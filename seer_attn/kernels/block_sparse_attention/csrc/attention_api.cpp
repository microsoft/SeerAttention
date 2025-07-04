#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_v2_cutlass", &flash_attention_v2_cutlass,
          "Flash attention v2 implement in cutlass");
    m.def("varlen_flash_attention_v2_cutlass", &varlen_flash_attention_v2_cutlass,
          "Varlen version of flash attention v2 implement in cutlass");          
    m.def("flash_attention_block_v2_cutlass", &flash_attention_block_v2_cutlass,
          "Flash attention block v2 implement in cutlass");
    m.def("varlen_flash_attention_block_v2_cutlass", &varlen_flash_attention_block_v2_cutlass,
          "Varlen version of flash attention block v2 implement in cutlass");          
}