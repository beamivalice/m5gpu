// m5gpu.c — Apple M5 GPU power-cap demo: set, burn, measure.
//
// Control:  IORegistryEntrySetCFProperties on AGXAccelerator with
//           { "SetMaxGPUAbsolutePower": true, "AbsoluteTarget": <mW> }
// Telemetry: MaxGPUAbsolutePower (cap as accepted), FilteredGPUPower (live mW)
// Work:     4096³ FP32 matmul burn, GPU-timestamp timed, live progress.
//
// Commands:
//   status            cap + live power readout (no root needed)
//   cap <watts>       set GPU power limit to <watts> (root)
//   sweep             GFLOPS at 5/10/15/20 W caps + uncapped (root)
//   burn [seconds]    full-speed burn with 500 ms telemetry (root)

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>

// ─── terminal ────────────────────────────────────────────────────────────────

static int g_color = 1;
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_CYAN    "\033[36m"
#define LOG_INFO(fmt, ...) printf("%s•%s " fmt "\n", g_color?C_CYAN:"", g_color?C_RESET:"", ##__VA_ARGS__)
#define LOG_OK(fmt, ...)   printf("%s✓%s " fmt "\n", g_color?C_GREEN:"", g_color?C_RESET:"", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) printf("%s⚠%s " fmt "\n", g_color?C_YELLOW:"", g_color?C_RESET:"", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...)  fprintf(stderr, "%s✗%s " fmt "\n", g_color?"\033[31m":"", g_color?C_RESET:"", ##__VA_ARGS__)

// ─── AGX power-cap interface ─────────────────────────────────────────────────

static io_service_t g_agx = 0;

static io_service_t get_agx(void) {
    if (g_agx) return g_agx;
    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault,
        IOServiceMatching("AGXAccelerator"), &iter);
    if (kr != KERN_SUCCESS) return 0;
    g_agx = IOIteratorNext(iter);
    IOObjectRelease(iter);
    return g_agx;
}

static int64_t agx_get(const char *key) {
    io_service_t svc = get_agx();
    if (!svc) return -1;
    CFStringRef k = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
    CFTypeRef v = IORegistryEntryCreateCFProperty(svc, k, NULL, 0);
    CFRelease(k);
    if (!v || CFGetTypeID(v) != CFNumberGetTypeID()) { if (v) CFRelease(v); return -1; }
    int64_t out = 0;
    CFNumberGetValue((CFNumberRef)v, kCFNumberSInt64Type, &out);
    CFRelease(v);
    return out;
}

/// Write { SetMaxGPUAbsolutePower: true, AbsoluteTarget: milliwatts }.
/// Returns the value that actually landed (negative = uncapped), or -1 on error.
static int64_t agx_set_power_cap(int64_t milliwatts) {
    io_service_t svc = get_agx();
    if (!svc) return -1;

    CFMutableDictionaryRef dict = CFDictionaryCreateMutable(
        NULL, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFStringRef k_trig = CFStringCreateWithCString(NULL, "SetMaxGPUAbsolutePower", kCFStringEncodingUTF8);
    CFStringRef k_tgt  = CFStringCreateWithCString(NULL, "AbsoluteTarget", kCFStringEncodingUTF8);
    CFDictionarySetValue(dict, k_trig, kCFBooleanTrue);
    CFNumberRef n = CFNumberCreate(NULL, kCFNumberSInt64Type, &milliwatts);
    CFDictionarySetValue(dict, k_tgt, n);

    kern_return_t kr = IORegistryEntrySetCFProperties(svc, dict);
    CFRelease(n); CFRelease(k_tgt); CFRelease(k_trig); CFRelease(dict);
    if (kr != KERN_SUCCESS) return -1;
    return agx_get("MaxGPUAbsolutePower");
}

// ─── Metal burn kernel ───────────────────────────────────────────────────────

static NSString *kShader = @
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void mm(device const float *A[[buffer(0)]],device const float *B[[buffer(1)]],"
"device float *C[[buffer(2)]],constant uint &N[[buffer(3)]],"
"uint2 g[[thread_position_in_grid]],uint2 l[[thread_position_in_threadgroup]]){"
"uint r=g.y,c=g.x;if(r>=N||c>=N)return;"
"threadgroup float As[16][16],Bs[16][16];float a=0;"
"for(uint t=0;t<(N+15)/16;t++){uint ac=t*16+l.x,br=t*16+l.y;"
"As[l.y][l.x]=(r<N&&ac<N)?A[r*N+ac]:0;Bs[l.y][l.x]=(br<N&&c<N)?B[br*N+c]:0;"
"threadgroup_barrier(mem_flags::mem_threadgroup);"
"for(uint i=0;i<16;i++)a=fma(As[l.y][i],Bs[i][l.x],a);"
"threadgroup_barrier(mem_flags::mem_threadgroup);}C[r*N+c]=a;}";

typedef struct {
    id<MTLDevice> dev;
    id<MTLComputePipelineState> pso;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> A, B, C, Nb;
    uint32_t N;
    double flop_per_pass;
} Burn;

static int burn_setup(Burn *b, uint32_t N) {
    memset(b, 0, sizeof(*b));
    b->dev = MTLCreateSystemDefaultDevice();
    if (!b->dev) return -1;
    NSError *err;
    id<MTLLibrary> lib = [b->dev newLibraryWithSource:kShader options:nil error:&err];
    if (!lib) return -1;
    b->pso = [b->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&err];
    b->queue = [b->dev newCommandQueue];
    b->N = N;
    size_t bytes = (size_t)N * N * 4;
    b->A = [b->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    b->B = [b->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    b->C = [b->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    b->Nb = [b->dev newBufferWithBytes:&N length:4 options:MTLResourceStorageModeShared];
    if (!b->pso || !b->queue || !b->A || !b->B || !b->C || !b->Nb) return -1;
    int32_t *ap = (int32_t *)b->A.contents;
    for (NSUInteger i = 0; i < (NSUInteger)N*N; i++) ap[i] = (int32_t)(i % 7) - 3;
    b->flop_per_pass = 2.0 * N * N * N;
    return 0;
}

static void burn_teardown(Burn *b) { memset(b, 0, sizeof(*b)); }

/// Runs one matmul pass; returns GPU time in ms via timestamps.
static double burn_pass(Burn *b) {
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:b->pso];
    [enc setBuffer:b->A offset:0 atIndex:0];
    [enc setBuffer:b->B offset:0 atIndex:1];
    [enc setBuffer:b->C offset:0 atIndex:2];
    [enc setBuffer:b->Nb offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(b->N, b->N, 1)
   threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    return ms > 0 ? ms : -1.0;
}

// ─── measurement loop with live sampling ────────────────────────────────────
static double now_s(void) {
    static mach_timebase_info_data_t tbi;
    if (!tbi.denom) mach_timebase_info(&tbi);
    return mach_absolute_time() * tbi.numer / tbi.denom / 1e9;
}

// ─── sampling ────────────────────────────────────────────────────────────────

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int _) { (void)_; g_stop = 1; }

typedef struct {
    double gflops;     // wall-clock average over the run
    double avg_watts;  // FilteredGPUPower averaged across the burn
} MeasResult;

/// Burn for `seconds`, discarding `warmup` seconds of clock/power transient
/// after a cap change. If `interval` > 0, print one live line every
/// `interval` seconds. Power is sampled every counted pass.
static MeasResult measure_loop(Burn *b, int seconds, double warmup,
                               double interval, const char *tag) {
    signal(SIGINT, on_sigint);
    MeasResult r = {0, 0};
    double started = now_s(), last_report = started;
    double window_start = started;
    double total_flops = 0, window_flops = 0;
    double mw_sum = 0;
    long mw_n = 0;

    while (!g_stop) {
        double t = now_s() - started;
        if (t >= seconds) break;
        burn_pass(b);
        if (t >= warmup) {   // count only post-transient passes
            total_flops += b->flop_per_pass;
            window_flops += b->flop_per_pass;
            int64_t mw = agx_get("FilteredGPUPower");
            if (mw > 0) { mw_sum += mw; mw_n++; }
        }

        double wall_now = now_s();
        if (interval > 0 && wall_now - last_report >= interval) {
            double win_wall = wall_now - window_start;
            printf("  [%s t=%5.1fs] %7.0f GFLOPS  %6.1f W\n",
                   tag, wall_now - started,
                   win_wall > 0 ? window_flops / (win_wall * 1e9) : 0,
                   mw_n > 0 ? mw_sum / mw_n / 1000.0 : 0);
            fflush(stdout);
            last_report = wall_now;
            window_start = wall_now;
            window_flops = 0;
        }
    }
    double counted_wall = (now_s() - started) - warmup;
    r.gflops = counted_wall > 0 ? total_flops / (counted_wall * 1e9) : 0;
    r.avg_watts = mw_n > 0 ? mw_sum / mw_n / 1000.0 : 0;
    return r;
}

// ─── concurrent frequency sampling (root) ────────────────────────────────────

#define PM_MAX_FREQ 1024

typedef struct {
    long med_mhz;      // median GPU HW active frequency
} PmStats;

static FILE *g_pm = NULL;
static long g_freqs[PM_MAX_FREQ];
static int g_nfreq = 0;

/// Start powermetrics in the background; it samples while the caller burns.
static void pm_start(int seconds) {
    char cmd[160];
    snprintf(cmd, sizeof cmd,
             "powermetrics --samplers gpu_power -i 400 -n %d 2>/dev/null",
             seconds * 3 + 4);
    g_pm = popen(cmd, "r");
    g_nfreq = 0;
}

static int cmp_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/// Wait for powermetrics to finish; return median sustained GPU MHz.
static PmStats pm_stop(void) {
    PmStats st = {-1};
    if (!g_pm) return st;
    char line[1024];
    const char *needle = "GPU HW active frequency:";
    while (fgets(line, sizeof line, g_pm)) {
        char *hit = strstr(line, needle);
        if (!hit || g_nfreq >= PM_MAX_FREQ) continue;
        long v = strtol(hit + strlen(needle), NULL, 10);
        if (v > 0 && v < 5000) g_freqs[g_nfreq++] = v;
    }
    pclose(g_pm);
    g_pm = NULL;

    if (g_nfreq > 0) {
        qsort(g_freqs, g_nfreq, sizeof(long), cmp_long);
        st.med_mhz = g_freqs[g_nfreq / 2];
    }
    return st;
}

// ─── commands ────────────────────────────────────────────────────────────────

/// Set the GPU power limit to <watts> and leave it active.
/// A positive value caps power at that level. A negative value removes the
/// limit entirely (written as -1000). Zero is refused — on M5 it means
/// "target zero watts" and parks the GPU at its frequency floor.
static void cmd_cap(int argc, char **argv) {
    if (argc != 3) {
        LOG_ERR("usage: sudo %s cap <watts>", getprogname());
        exit(1);
    }
    long long watts = -1;
    if (strcmp(argv[2], "off") != 0 && strcmp(argv[2], "uncap") != 0) {
        const char *digits = argv[2];
        if (*digits == '-') digits++;
        if (!*digits || strspn(digits, "0123456789") != strlen(digits)) {
            LOG_ERR("cap must be a positive integer wattage or 'off'");
            exit(1);
        }
        errno = 0;
        char *end = NULL;
        watts = strtoll(argv[2], &end, 10);
        if (errno == ERANGE || *end || watts > INT64_MAX / 1000 || watts == 0) {
            LOG_ERR("invalid wattage; zero parks the GPU. Use a positive integer or 'off'");
            exit(1);
        }
    }
    if (geteuid() != 0) { LOG_ERR("cap needs root, hint: sudo ./m5gpu cap 40"); exit(1); }
    if (!get_agx()) { LOG_ERR("AGXAccelerator not found"); exit(1); }
    int64_t want = watts < 0 ? -1000 : (int64_t)watts * 1000;

    int64_t got = agx_set_power_cap(want);
    if (got != want) {
        LOG_ERR("cap did not land (driver accepted %lld, wanted %lld)", got, want);
        exit(1);
    }
    if (watts < 0)
        LOG_OK("power limit removed — full boost restored");
    else
        LOG_OK("GPU capped at %lld W", watts);
}

static void cmd_status(int json) {
    io_service_t svc = get_agx();
    if (!svc) { LOG_ERR("AGXAccelerator not found"); exit(1); }
    int64_t cap = agx_get("MaxGPUAbsolutePower");
    int64_t mw = agx_get("FilteredGPUPower");
    if (json) {
        if (cap == -1 || mw < 0) {
            LOG_ERR("GPU power properties unavailable");
            exit(1);
        }
        printf("{\"cap_milliwatts\":%lld,\"cap_watts\":", cap);
        if (cap >= 0) printf("%.3f", cap / 1000.0);
        else printf("null");
        printf(",\"capped\":%s,\"draw_watts\":%.3f}\n",
               cap >= 0 ? "true" : "false", mw / 1000.0);
        return; // telemetry only; do not run a benchmark for MCP polling
    }
    printf("%sGPU power status%s\n", g_color?C_BOLD:"", g_color?C_RESET:"");
    if (cap > 0) printf("  cap:      %lld W\n", cap / 1000);
    else          printf("  cap:      none (0)\n");
    printf("  draw:     %.3f W (filtered)\n", mw > 0 ? mw / 1000.0 : 0.0);

    // quick 1-second speed probe
    Burn b;
    if (burn_setup(&b, 2048) == 0) {
        double best_ms = 1e9;
        for (int i = 0; i < 5; i++) {
            double ms = burn_pass(&b);
            if (ms > 0 && ms < best_ms) best_ms = ms;
        }
        printf("  compute:  %.0f GFLOPS (2048³ matmul best-of-5)\n",
               b.flop_per_pass / (best_ms / 1000.0) / 1e9);
        burn_teardown(&b);
    }
}

static void cmd_burn(int seconds) {
    if (geteuid() != 0) { LOG_ERR("burn needs root, hint: sudo ./m5gpu sweep"); exit(1); }
    Burn b;
    if (burn_setup(&b, 4096) != 0) { LOG_ERR("Metal setup failed"); exit(1); }
    LOG_INFO("burning 4096³ matmul for %d s — ctrl-c to stop early", seconds);
    MeasResult m = measure_loop(&b, seconds, 0, 0.5, "burn");
    LOG_OK("average: %.0f GFLOPS at %.2f W", m.gflops, m.avg_watts);
    burn_teardown(&b);
}

static void cmd_sweep(int argc, char **argv) {
    if (geteuid() != 0) { LOG_ERR("sweep needs root, hint: sudo ./m5gpu sweep"); exit(1); }

    int caps[32];
    int n_caps = 0;
    if (argc > 2) {
        for (int i = 2; i < argc && n_caps < 32; i++)
            caps[n_caps++] = atoi(argv[i]);
    } else {
        const int defaults[] = {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100};
        for (int i = 0; i < (int)(sizeof(defaults)/sizeof(defaults[0])); i++)
            caps[n_caps++] = defaults[i];
    }
    const int seconds_per_cap = 5;

    printf("\n  %s%-6s %-10s %-9s %-10s %s%s\n", g_color?C_BOLD:"", 
           "cap", "freq", "power", "GFLOPS", "GF/W", g_color?C_RESET:"");
    printf("  %s------------------------------------------------------------%s\n",
           g_color?C_DIM:"", g_color?C_RESET:"");

    for (int i = 0; i < n_caps; i++) {
        int64_t want = (int64_t)caps[i] * 1000;
        if (agx_set_power_cap(want) != want) {
            LOG_WARN("cap %d W did not land", caps[i]);
            continue;
        }
        sleep(4); // CLPC settle

        Burn b;
        if (burn_setup(&b, 4096) != 0) { LOG_ERR("Metal setup failed"); continue; }
        pm_start(seconds_per_cap);
        MeasResult m = measure_loop(&b, seconds_per_cap + 2.0, 2.0, 0, NULL); // quiet
        PmStats st = pm_stop();
        double eff = m.avg_watts > 0 ? m.gflops / m.avg_watts : 0;
        char freq_buf[16];
        snprintf(freq_buf, sizeof freq_buf, "%ld MHz", st.med_mhz);
        printf("  %-6d %-10s %6.2f W  %8.0f   %5.1f\n",
               caps[i], freq_buf, m.avg_watts, m.gflops, eff);
        fflush(stdout);
        burn_teardown(&b);
    }

    // Uncapped reference: -1 disables the cap entirely (0 would park the
    // GPU at its frequency floor — never write 0, see README).
    agx_set_power_cap(-1000);
    sleep(4);
    Burn b;
    if (burn_setup(&b, 4096) == 0) {
        pm_start(seconds_per_cap);
        MeasResult m = measure_loop(&b, seconds_per_cap + 2.0, 2.0, 0, NULL);
        PmStats st = pm_stop();
        double eff = m.avg_watts > 0 ? m.gflops / m.avg_watts : 0;
        char freq_buf[16];
        snprintf(freq_buf, sizeof freq_buf, "%ld MHz", st.med_mhz);
        printf("  %-6s %-10s %6.2f W  %8.0f   %5.1f\n",
               "uncap", freq_buf, m.avg_watts, m.gflops, eff);
        burn_teardown(&b);
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

static void usage(const char *prog) {
    printf("usage:\n");
    printf("  %s status            cap + live power (no root)\n", prog);
    printf("  %s status --json     JSON telemetry without a benchmark\n", prog);
    printf("  sudo %s cap <watts>  set GPU power limit to <watts>\n", prog);
    printf("  sudo %s cap off      remove the power limit (restores boost)\n", prog);
    printf("  sudo %s sweep        GFLOPS at 5..100 W caps + uncapped\n", prog);
    printf("  sudo %s burn [sec]   full-speed burn w/ live telemetry\n", prog);
}

int main(int argc, char **argv) {
    if (!isatty(STDOUT_FILENO)) g_color = 0;
    const char *cmd = argc > 1 ? argv[1] : "status";

    if (strcmp(cmd, "status") == 0) {
        if (argc > 3 || (argc == 3 && strcmp(argv[2], "--json") != 0)) {
            usage(argv[0]);
            return 1;
        }
        cmd_status(argc == 3);
    }
    else if (strcmp(cmd, "cap") == 0) cmd_cap(argc, argv);
    else if (strcmp(cmd, "sweep") == 0) cmd_sweep(argc, argv);
    else if (strcmp(cmd, "burn") == 0) cmd_burn(argc > 2 ? atoi(argv[2]) : 20);
    else { usage(argv[0]); return 1; }
    return 0;
}
