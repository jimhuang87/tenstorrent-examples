#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include <iostream>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // 1. 初始化裝置 (Device 0)
    int device_id = 0;
    Device *device = CreateDevice(device_id);

    // 2. 建立 Program
    Program program = CreateProgram();
    CoreCoord core = {0, 0}; // 簡單起見，只用一個核心 (0,0)

    // 3. 設定參數
    uint32_t num_tiles = 32;
    uint32_t tile_size = 2048; // 32x32 bfloat16
    uint32_t total_size = num_tiles * tile_size;

    // 4. 建立 DRAM Buffers (Input & Output)
    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device=device,
        .size = total_size,
        .page_size = tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    std::shared_ptr<Buffer> src_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst_buffer = CreateBuffer(dram_config);

    // 5. 設定 Circular Buffers (L1 通道)
    // Input CB
    CircularBufferConfig cb_src_config = CircularBufferConfig(tile_size * 2, {{tt::CB::c_in0, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CB::c_in0, tile_size);
    CreateCircularBuffer(program, core, cb_src_config);

    // Output CB
    CircularBufferConfig cb_dst_config = CircularBufferConfig(tile_size * 2, {{tt::CB::c_out0, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CB::c_out0, tile_size);
    CreateCircularBuffer(program, core, cb_dst_config);

    // 6. 建立 Kernels
    // Reader
    std::vector<uint32_t> reader_args = {
        src_buffer->address(),
        (uint32_t)src_buffer->noc_coordinates().x,
        (uint32_t)src_buffer->noc_coordinates().y,
        num_tiles
    };
    CreateKernel(program, "src/reader_kernel.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    
    // Writer
    std::vector<uint32_t> writer_args = {
        dst_buffer->address(),
        (uint32_t)dst_buffer->noc_coordinates().x,
        (uint32_t)dst_buffer->noc_coordinates().y,
        num_tiles
    };
    CreateKernel(program, "src/writer_kernel.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute
    std::vector<uint32_t> compute_args = {num_tiles};
    CreateKernel(program, "src/add_one_compute.cpp", core, ComputeConfig{.math_approx_mode = false, .compile_args = compute_args});

    // 7. 寫入 Runtime Args
    SetRuntimeArgs(program, core, GetKernel(program, "src/reader_kernel.cpp"), reader_args);
    SetRuntimeArgs(program, core, GetKernel(program, "src/writer_kernel.cpp"), writer_args);
    SetRuntimeArgs(program, core, GetKernel(program, "src/add_one_compute.cpp"), compute_args);

    // 8. 準備數據並寫入裝置 (Host -> Device)
    std::vector<uint32_t> host_data(total_size / sizeof(uint32_t), 0x3f803f80); // bfloat16 of 1.0 packed
    WriteToBuffer(src_buffer, host_data);

    // 9. 執行 (Launch!)
    std::cout << "Starting Device..." << std::endl;
    Detail::LaunchProgram(device, program);
    std::cout << "Device Finished." << std::endl;

    // 10. 讀回結果
    std::vector<uint32_t> result_data;
    ReadFromBuffer(dst_buffer, result_data);

    std::cout << "Success! First value: " << std::hex << result_data[0] << std::endl;

    // 關閉
    CloseDevice(device);
    return 0;
}