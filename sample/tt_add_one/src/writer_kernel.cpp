#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(1);
    uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(2);
    uint32_t num_tiles             = get_arg_val<uint32_t>(3);

    constexpr uint32_t tile_size_bytes = get_tile_size(tt::CB::c_out0);
    const uint32_t cb_id_out = tt::CB::c_out0;

    uint64_t dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 1. 等待 Compute Kernel 算完並放入 Buffer
        cb_wait_front(cb_id_out, 1);
        
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // 2. 寫回 DRAM
        noc_async_write(l1_read_addr, dst_noc_addr, tile_size_bytes);
        noc_async_write_barrier();

        // 3. 釋放 Buffer 空間
        cb_pop_front(cb_id_out, 1);

        dst_noc_addr += tile_size_bytes;
    }
}