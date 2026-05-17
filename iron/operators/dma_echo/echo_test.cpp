// echo_test.cpp — Standalone DMA echo test via custom XRT dispatch.
// Bypasses IRON framework to test raw DMA path on XDNA NPU.
//
// Usage: echo_test.exe <xclbin_path> <insts_path> [version]
//   version: 1 = single ObjectFifo copy (default)
//            2 = dual ObjectFifo concat

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_xclbin.h"
#include "xrt/xrt_hw_context.h"
#include "experimental/xrt_ext.h"

static std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        exit(1);
    }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint16_t bf = (uint16_t)(bits >> 16);
    if ((bits & 0x7FFFFFFF) < 0x7F800000) {
        uint32_t rounding = 0x7FFF + ((bf & 1) ^ 1);
        bf = (uint16_t)((bits + rounding) >> 16);
    }
    return bf;
}

static float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char * argv[]) {
    if (argc < 3) {
        printf("Usage: %s <xclbin> <insts> [version]\n", argv[0]);
        printf("  version: 1=copy (default), 2=concat\n");
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string insts_path = argv[2];
    int version = (argc >= 4) ? atoi(argv[3]) : 1;
    int N = 256;

    printf("echo_test v%d: xclbin=%s insts=%s N=%d\n",
           version, xclbin_path.c_str(), insts_path.c_str(), N);

    // --- Init XRT ---
    xrt::device dev(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    dev.register_xclbin(xclbin);
    auto uuid = xclbin.get_uuid();
    xrt::hw_context hw(dev, uuid);
    xrt::kernel kernel(hw, "MLIR_AIE");

    // Print group_ids
    printf("Kernel group_ids:");
    for (int a = 0; a < 10; a++) {
        try { printf(" [%d]=%zu", a, (size_t)kernel.group_id(a)); }
        catch (...) { printf(" [%d]=ERR", a); break; }
    }
    printf("\n");

    // --- Load insts ---
    auto insts_data = read_file(insts_path);
    printf("insts: %zu bytes\n", insts_data.size());

    xrt::bo insts_bo(dev, insts_data.size(),
                     xrt::bo::flags::cacheable, kernel.group_id(1));
    memcpy(insts_bo.map<void*>(), insts_data.data(), insts_data.size());
    insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("insts_bo synced to device\n");

    if (version == 1) {
        // --- Echo v1: single copy ---
        int buf_bytes = N * 2;  // bf16 = 2 bytes

        xrt::bo bo_in(dev, buf_bytes, xrt::bo::flags::host_only, kernel.group_id(3));
        xrt::bo bo_out(dev, buf_bytes, xrt::bo::flags::host_only, kernel.group_id(4));

        // Fill input with known bf16 pattern
        auto in_ptr = bo_in.map<uint16_t*>();
        for (int i = 0; i < N; i++) {
            float f = (float)(i + 1) * 0.5f;  // 0.5, 1.0, 1.5, ...
            in_ptr[i] = f32_to_bf16(f);
        }
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        printf("Input synced: ");
        for (int i = 0; i < 8; i++) printf("0x%04X ", in_ptr[i]);
        printf("...\n");

        // Clear output
        auto out_ptr = bo_out.map<uint16_t*>();
        memset(out_ptr, 0, buf_bytes);

        // Dispatch
        printf("Dispatching kernel...\n");
        auto run = xrt::run(kernel);
        run.set_arg(0, 3u);  // opcode
        run.set_arg(1, insts_bo);
        run.set_arg(2, (uint32_t)insts_data.size());
        run.set_arg(3, bo_in);
        run.set_arg(4, bo_out);

        auto state = run.wait(10000);  // 10s timeout

        if (state != ERT_CMD_STATE_COMPLETED) {
            printf("FAIL: kernel returned state=%d\n", (int)state);
            return 1;
        }
        printf("Kernel completed\n");

        // Verify
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        printf("Output:    ");
        for (int i = 0; i < 8; i++) printf("0x%04X ", out_ptr[i]);
        printf("...\n");

        int match = 0;
        for (int i = 0; i < N; i++) {
            if (out_ptr[i] == in_ptr[i]) match++;
        }
        printf("Result: %d/%d match\n", match, N);

        if (match != N) {
            printf("FAIL: output mismatch\n");
            printf("First 8 diffs:\n");
            for (int i = 0; i < 8; i++) {
                printf("  [%d] in=0x%04X out=0x%04X (%.4f vs %.4f)\n",
                       i, in_ptr[i], out_ptr[i],
                       bf16_to_f32(in_ptr[i]), bf16_to_f32(out_ptr[i]));
            }
            return 1;
        }
        printf("PASS: echo v1\n");

    } else {
        // --- Echo v2: dual concat ---
        int half_bytes = N * 2;
        int full_bytes = 2 * N * 2;

        xrt::bo bo_a(dev, half_bytes, xrt::bo::flags::host_only, kernel.group_id(3));
        xrt::bo bo_b(dev, half_bytes, xrt::bo::flags::host_only, kernel.group_id(4));
        xrt::bo bo_out(dev, full_bytes, xrt::bo::flags::host_only, kernel.group_id(5));

        // Fill A and B with different patterns
        auto a_ptr = bo_a.map<uint16_t*>();
        auto b_ptr = bo_b.map<uint16_t*>();
        for (int i = 0; i < N; i++) {
            a_ptr[i] = f32_to_bf16((float)(i + 1));       // 1.0, 2.0, 3.0, ...
            b_ptr[i] = f32_to_bf16((float)(i + 1) * 10);  // 10, 20, 30, ...
        }
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        printf("Inputs synced\n");

        // Clear output
        auto out_ptr = bo_out.map<uint16_t*>();
        memset(out_ptr, 0, full_bytes);

        // Dispatch
        printf("Dispatching kernel...\n");
        auto run = xrt::run(kernel);
        run.set_arg(0, 3u);
        run.set_arg(1, insts_bo);
        run.set_arg(2, (uint32_t)insts_data.size());
        run.set_arg(3, bo_a);
        run.set_arg(4, bo_b);
        run.set_arg(5, bo_out);

        auto state = run.wait(10000);

        if (state != ERT_CMD_STATE_COMPLETED) {
            printf("FAIL: kernel returned state=%d\n", (int)state);
            return 1;
        }
        printf("Kernel completed\n");

        // Verify: out[0..N-1] should match A, out[N..2N-1] should match B
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        int match_a = 0, match_b = 0;
        for (int i = 0; i < N; i++) {
            if (out_ptr[i] == a_ptr[i]) match_a++;
            if (out_ptr[N + i] == b_ptr[i]) match_b++;
        }
        printf("Result: A=%d/%d B=%d/%d\n", match_a, N, match_b, N);

        if (match_a != N || match_b != N) {
            printf("FAIL: output mismatch\n");
            if (match_a != N) {
                printf("First 8 A diffs:\n");
                for (int i = 0; i < 8; i++) {
                    printf("  [%d] a=0x%04X out=0x%04X\n", i, a_ptr[i], out_ptr[i]);
                }
            }
            if (match_b != N) {
                printf("First 8 B diffs:\n");
                for (int i = 0; i < 8; i++) {
                    printf("  [%d] b=0x%04X out=0x%04X\n", i, b_ptr[i], out_ptr[N + i]);
                }
            }
            return 1;
        }
        printf("PASS: echo v2\n");
    }

    return 0;
}
