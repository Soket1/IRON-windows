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
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_ext.h"

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
    fprintf(stderr, "[echo_test] argc=%d\n", argc); fflush(stderr);
    if (argc < 3) {
        fprintf(stderr, "[echo_test] usage branch\n"); fflush(stderr);
        printf("Usage: %s <xclbin> <insts> [version]\n", argv[0]);
        printf("  version: 1=copy (default), 2=concat\n");
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string insts_path = argv[2];
    fprintf(stderr, "[echo_test] paths copied\n"); fflush(stderr);
    int version = (argc >= 4) ? atoi(argv[3]) : 1;
    fprintf(stderr, "[echo_test] version=%d\n", version); fflush(stderr);
    int N = 256;

    printf("echo_test v%d: xclbin=%s insts=%s N=%d\n",
           version, xclbin_path.c_str(), insts_path.c_str(), N);
    fflush(stdout);
    fprintf(stderr, "[echo_test] before xrt::device\n"); fflush(stderr);

    // --- Init XRT ---
    xrt::device dev(0);
    fprintf(stderr, "[echo_test] after xrt::device\n"); fflush(stderr);
    auto xclbin = xrt::xclbin(xclbin_path);
    fprintf(stderr, "[echo_test] after xrt::xclbin\n"); fflush(stderr);
    dev.register_xclbin(xclbin);
    fprintf(stderr, "[echo_test] after register_xclbin\n"); fflush(stderr);
    auto uuid = xclbin.get_uuid();
    fprintf(stderr, "[echo_test] after get_uuid\n"); fflush(stderr);
    xrt::hw_context hw(dev, uuid);
    fprintf(stderr, "[echo_test] after hw_context\n"); fflush(stderr);
    xrt::kernel kernel(hw, "MLIR_AIE");
    fprintf(stderr, "[echo_test] after kernel\n"); fflush(stderr);

    // Print group_ids
    printf("Kernel group_ids:");
    for (int a = 0; a < 10; a++) {
        try { printf(" [%d]=%zu", a, (size_t)kernel.group_id(a)); fflush(stdout); }
        catch (...) { printf(" [%d]=ERR", a); fflush(stdout); break; }
    }
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "[echo_test] after group_ids\n"); fflush(stderr);

    // --- Load insts ---
    auto insts_data = read_file(insts_path);
    fprintf(stderr, "[echo_test] after read_file\n"); fflush(stderr);
    printf("insts: %zu bytes\n", insts_data.size());
    fflush(stdout);

    xrt::bo insts_bo(dev, insts_data.size(),
                     xrt::bo::flags::cacheable, kernel.group_id(1));
    fprintf(stderr, "[echo_test] after insts_bo create\n"); fflush(stderr);
    memcpy(insts_bo.map<void*>(), insts_data.data(), insts_data.size());
    fprintf(stderr, "[echo_test] after insts_bo map/memcpy\n"); fflush(stderr);
    insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[echo_test] after insts_bo sync\n"); fflush(stderr);
    printf("insts_bo synced to device\n");
    fflush(stdout);

    if (version == 1) {
        fprintf(stderr, "[echo_test] enter v1 branch\n"); fflush(stderr);
        // --- Echo v1: single copy ---
        int buf_bytes = N * 2;  // bf16 = 2 bytes

        xrt::bo bo_in(dev, buf_bytes, xrt::bo::flags::host_only, kernel.group_id(3));
        fprintf(stderr, "[echo_test] after bo_in create\n"); fflush(stderr);
        xrt::bo bo_out(dev, buf_bytes, xrt::bo::flags::host_only, kernel.group_id(4));
        fprintf(stderr, "[echo_test] after bo_out create\n"); fflush(stderr);

        // Fill input with known bf16 pattern
        auto in_ptr = bo_in.map<uint16_t*>();
        fprintf(stderr, "[echo_test] after bo_in map\n"); fflush(stderr);
        for (int i = 0; i < N; i++) {
            float f = (float)(i + 1) * 0.5f;  // 0.5, 1.0, 1.5, ...
            in_ptr[i] = f32_to_bf16(f);
        }
        fprintf(stderr, "[echo_test] after fill input\n"); fflush(stderr);
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fprintf(stderr, "[echo_test] after bo_in sync\n"); fflush(stderr);
        printf("Input synced: ");
        for (int i = 0; i < 8; i++) printf("0x%04X ", in_ptr[i]);
        printf("...\n");
        fflush(stdout);

        // Clear output
        auto out_ptr = bo_out.map<uint16_t*>();
        fprintf(stderr, "[echo_test] after bo_out map\n"); fflush(stderr);
        memset(out_ptr, 0, buf_bytes);
        fprintf(stderr, "[echo_test] after clear output\n"); fflush(stderr);

        // Dispatch
        printf("Dispatching kernel...\n");
        fflush(stdout);
        fprintf(stderr, "[echo_test] before run create\n"); fflush(stderr);
        auto run = xrt::run(kernel);
        fprintf(stderr, "[echo_test] after run create\n"); fflush(stderr);
        run.set_arg(0, 3u);  // opcode
        fprintf(stderr, "[echo_test] after set_arg0\n"); fflush(stderr);
        run.set_arg(1, insts_bo);
        fprintf(stderr, "[echo_test] after set_arg1\n"); fflush(stderr);
        run.set_arg(2, (uint32_t)insts_data.size());
        fprintf(stderr, "[echo_test] after set_arg2\n"); fflush(stderr);
        run.set_arg(3, bo_in);
        fprintf(stderr, "[echo_test] after set_arg3\n"); fflush(stderr);
        run.set_arg(4, bo_out);
        fprintf(stderr, "[echo_test] after set_arg4\n"); fflush(stderr);

        fprintf(stderr, "[echo_test] before start\n"); fflush(stderr);
        run.start();
        fprintf(stderr, "[echo_test] after start\n"); fflush(stderr);
        auto state = run.wait(10000);  // 10s timeout
        fprintf(stderr, "[echo_test] after wait state=%d\n", (int)state); fflush(stderr);

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
