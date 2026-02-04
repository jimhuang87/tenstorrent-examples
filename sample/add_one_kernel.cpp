// 這是跑在 Device 上的 "Compute Kernel"
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    // 1. 初始化數學運算單元
    binary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);

    // 2. 獲取這個 Core 要處理幾個 Tiles
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    // 3. 迴圈處理每一個 Tile
    for(uint32_t i = 0; i < per_core_tile_cnt; ++i) {
        // 從輸入 Buffer 拿一個 Tile
        acquire_dst(tt::CB::c_in0); 

        // 執行 "加 1" 運算 (這是一個 Composite Op 或更底層的指令)
        // 這裡簡化示意，實際上可能呼叫 add_tiles
        add_unary_tile_init();
        pack_tile(0, tt::CB::c_out0);

        // 釋放 Buffer
        release_dst(tt::CB::c_out0);
    }
}
}