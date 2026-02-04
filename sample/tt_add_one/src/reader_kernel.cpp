#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // 參數由 Host 傳入
    uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(0);
    uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(1);
    uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(2);
    uint32_t num_tiles             = get_arg_val<uint32_t>(3);

    // 每個 Tile 大小 (32x32 bfloat16 = 2048 bytes)
    constexpr uint32_t tile_size_bytes = get_tile_size(tt::CB::c_in0);
    constexpr uint32_t data_format_bytes = 2; // bfloat16

    // 設定 Circular Buffer 介面
    const uint32_t cb_id_in = tt::CB::c_in0;

    // 建立 DRAM 讀取器物件
    // 注意：這裡簡化了多 Bank 讀取，假設是單一 Bank 連續讀取
    uint64_t src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 1. 預留 L1 空間
        cb_reserve_back(cb_id_in, 1);
        
        // 2. 獲取 L1 寫入指標
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);

        // 3. 從 DRAM 非同步讀取到 L1
        noc_async_read(src_noc_addr, l1_write_addr, tile_size_bytes);
        
        // 4. 等待讀取完成 (這是一個 Barrier)
        noc_async_read_barrier();

        // 5. 推送數據給 Compute Kernel
        cb_push_back(cb_id_in, 1);

        // 移動 DRAM 指標到下一個 Tile
        src_noc_addr += tile_size_bytes;
    }
}