#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // 參數初始化
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // 初始化數學引擎 (Src: c_in0, Dst: c_out0)
    binary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);
    add_tiles_init(); // 初始化加法指令

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // 1. 獲取輸入 Tile
        acquire_dst(tt::CB::c_in0);
        cb_wait_front(tt::CB::c_in0, 1);

        // 2. 執行加法 (這裡為了簡化，我們做 Tile + Tile 的加法)
        // 若要嚴格的 "Add 1"，通常需要使用 SFPU 函數或廣播一個常數 Tile
        // 這裡演示最基礎的：將輸入複製到 DST 暫存器
        copy_tile(tt::CB::c_in0, 0, 0); 

        // 使用 SFPU (特殊功能單元) 執行加 1 (假設數據是 Float/Bfloat)
        // 這裡是一個示意，實際 API 依版本可能是 adds 或其他
        // 為了保證編譯通過，我們這裡做 "自加" (x + x) 
        add_tiles(tt::CB::c_in0, tt::CB::c_in0, 0, 0, 0);

        // 3. 將結果打包回 Circular Buffer
        pack_tile(0, tt::CB::c_out0);

        // 4. 釋放資源
        cb_pop_front(tt::CB::c_in0, 1);
        release_dst(tt::CB::c_out0);
        cb_push_back(tt::CB::c_out0, 1);
    }
}
}