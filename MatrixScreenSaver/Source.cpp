// MatrixScreensaver.cpp
// A highly advanced, GPU-accelerated Windows screensaver (.scr) implementing a
// falling-katakana "Matrix rain" canvas layered with an integrated real-time 
// hardware monitoring telemetry matrix and device-level history sparklines.
//
// Features:
// - Explicit Dedicated GPU enumeration preference over Integrated Devices via DXGI.
// - Localized metrics separation (Program Footprint vs Total Device Metrics).
// - Power, Voltage, and Core thermal monitoring arrays.
// - Hardware status sparklines reflecting system-wide performance conditions.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <windows.h>
#include <psapi.h>
#include <d2d1.h>
#include <dxgi1_4.h> // Enhanced DXGI interfaces for memory queries
#include <dwrite.h>
#include <pdh.h>
#include <pdhmsg.h>   
#include <wrl/client.h> 

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "Pdh.lib")
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Configuration / Windowing Defaults
// ---------------------------------------------------------------------------
static const int  MOVE_THRESHOLD = 8;
static const UINT TIMER_ID = 1;
static const UINT TIMER_INTERVAL_MS = 16; // Stable frame pump step

static const int KATAKANA_COUNT = 96;
static const int NUM_LAYERS = 4;

struct LayerConfig {
    int   fontSize;
    float speedMul;
    float brightnessMul;
    int   columnOffsetPx;
    float columnDensity;
    float trailR, trailG, trailB;
    float headR, headG, headB;
};

static const LayerConfig LAYER_CONFIGS[NUM_LAYERS] = {
    { 8,  0.35f, 0.16f, 3,  0.35f,  20.0f, 110.0f, 90.0f,   140.0f, 220.0f, 200.0f },
    { 11, 0.55f, 0.30f, 9,  0.45f,  25.0f, 150.0f, 80.0f,   160.0f, 240.0f, 180.0f },
    { 14, 0.78f, 0.58f, 5,  0.55f,  35.0f, 180.0f, 40.0f,   180.0f, 255.0f, 150.0f },
    { 18, 1.00f, 1.00f, 0,  0.65f,  60.0f, 220.0f, 20.0f,   210.0f, 255.0f, 190.0f },
};

struct Symbol {
    float   y = 0.0f;
    int     speed = 0;
    wchar_t value = L' ';
    int     interval = 0;
    bool    isHead = false;
};

struct ColumnState {
    int x = 0;
    int layer = 0;
    int speed = 0;
    int length = 0;
    std::vector<Symbol> symbols;
};

static HWND  g_hwnd = nullptr;
static int   g_width = 0, g_height = 0;
static std::vector<ColumnState> g_columns;
// Column indices bucketed per layer so DrawFrame can iterate each column
// exactly once per frame instead of scanning the full column list once
// per layer (was O(NUM_LAYERS * columns), now O(columns)).
static std::vector<std::vector<size_t>> g_columnsByLayer;

// ---------------------------------------------------------------------------
// Direct2D / Pipeline Graphics Core
// ---------------------------------------------------------------------------
static ComPtr<ID2D1Factory>          g_d2dFactory;
static ComPtr<ID2D1HwndRenderTarget> g_renderTarget;
static ComPtr<IDWriteFactory>        g_dwriteFactory;
static ComPtr<IDWriteTextFormat>     g_overlayTextFormat;
static ComPtr<ID2D1SolidColorBrush>  g_fadeBrush;
static ComPtr<ID2D1SolidColorBrush>  g_overlayBrush;
static ComPtr<ID2D1SolidColorBrush>  g_sparklineStrokeBrush;
static ComPtr<ID2D1SolidColorBrush>  g_sparklineFillBrush;
static ComPtr<ID2D1SolidColorBrush>  g_overlayBgBrush;
static ComPtr<ID2D1SolidColorBrush>  g_overlayBorderBrush;
static ComPtr<ID2D1BitmapRenderTarget> g_trailTarget;

struct GlyphAtlas {
    ComPtr<ID2D1Bitmap> headBitmap;
    ComPtr<ID2D1Bitmap> trailBitmap;
    int   cellWidth = 0;
    int   cellHeight = 0;
    int   glyphCount = 0;
};
static GlyphAtlas g_atlases[NUM_LAYERS];

static bool g_isPreview = false;
static bool g_benchmarkMode = false;
static ULONGLONG g_startTick = 0;
static const ULONGLONG STARTUP_GRACE_MS = 1000;
static bool InGracePeriod() { return (GetTickCount64() - g_startTick) < STARTUP_GRACE_MS; }

static POINT g_lastMousePos{};
static bool  g_mouseInit = false;

// ---------------------------------------------------------------------------
// Telemetry Tracking & Ring Buffers
// ---------------------------------------------------------------------------
static const int SPARKLINE_HISTORY_SIZE = 60;

struct PerformanceHistory {
    float data[SPARKLINE_HISTORY_SIZE] = { 0.0f };
    int head = 0;

    void Push(float val) {
        data[head] = val;
        head = (head + 1) % SPARKLINE_HISTORY_SIZE;
    }
    float GetAt(int idx) const {
        return data[(head + idx) % SPARKLINE_HISTORY_SIZE];
    }
};

struct HardwareDiagnostics {
    // String descriptions & Architectural classifications
    std::wstring gpuDeviceName = L"Processing Execution Target...";
    std::wstring gpuArchitectureType = L"Discrete / Dedicated High Performance";
    std::wstring gpuDriverCode = L"N/A";

    // Program Isolation Metrics
    double programCpuPercent = 0.0;
    double programRamUsedMB = 0.0;
    double programVramUsedMB = 0.0;

    // Total System Metrics
    double systemFps = 0.0;
    double systemCpuPercent = 0.0;
    double systemGpuPercent = 0.0;
    int    systemRamPercent = 0;
    double systemRamUsedGB = 0.0;
    double systemRamTotalGB = 0.0;

    // Environmental Tracking Matrices
    float  cpuCoreVoltage = 1.21f;
    float  gpuCoreVoltage = 0.98f;
    float  cpuTemperature = 48.0f;
    float  gpuTemperature = 54.0f;

    // Battery Status Substructures
    bool   hasBattery = false;
    bool   batteryCharging = false;
    int    batteryPercent = 100;
    DWORD  batterySecondsLeft = 0;

    // Device-wide Trace Graphs
    PerformanceHistory totalFpsHistory;
    PerformanceHistory totalCpuHistory;
    PerformanceHistory totalGpuHistory;
    PerformanceHistory totalRamHistory;
};
static HardwareDiagnostics g_hw;

static ULONGLONG g_fpsWindowStart = 0;
static int       g_fpsFrameCount = 0;
static ULONGLONG g_lastCpuCheckTick = 0;
static ULONGLONG g_lastKernel100ns = 0, g_lastUser100ns = 0;
static ULONGLONG g_lastSysKernel100ns = 0, g_lastSysUser100ns = 0, g_lastSysIdle100ns = 0;
static int       g_numCores = 1;

static PDH_HQUERY   g_pdhQuery = nullptr;
static PDH_HCOUNTER g_pdhGpuCounter = nullptr;
static bool         g_pdhAvailable = false;
static ComPtr<IDXGIAdapter3> g_selectedAdapter; // Active Selected Render Loop Device

// ---------------------------------------------------------------------------
// Matrix Cascade Random String Generation Engine
// ---------------------------------------------------------------------------
// MSVC's rand() has a period of only RAND_MAX (32767) and involves a global
// lock, which is wasteful given how frequently this is called (every glyph,
// every few frames, across every column). A small xorshift32 PRNG is faster,
// branch-free, and has a far longer period; it's used purely for cosmetic
// randomness so it doesn't need to be cryptographically strong.
static uint32_t g_rngState = 0x9E3779B9u;

static void SeedFastRng(uint32_t seed) {
    g_rngState = seed ? seed : 0x9E3779B9u;
}

static uint32_t FastRandU32() {
    uint32_t x = g_rngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rngState = x;
    return x;
}

// Returns a value in the half-open range [0, bound), using a widening
// multiply instead of modulo to avoid the low-bit periodicity issues
// xorshift generators can have with the '%' operator.
static uint32_t FastRandBounded(uint32_t bound) {
    return static_cast<uint32_t>((static_cast<uint64_t>(FastRandU32()) * bound) >> 32);
}

static float FastRandFloat01() {
    return static_cast<float>(FastRandU32() >> 8) / static_cast<float>(1u << 24);
}

static wchar_t RandomKatakana() {
    return static_cast<wchar_t>(0x30A0 + FastRandBounded(KATAKANA_COUNT));
}

// ---------------------------------------------------------------------------
// Hardware Interrogation & Processing Selection Engines
// ---------------------------------------------------------------------------
static void SelectPreferredGraphicsDevice() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        g_hw.gpuDeviceName = L"Default Graphics Context";
        return;
    }

    ComPtr<IDXGIAdapter1> adapter1;
    ComPtr<IDXGIAdapter1> bestAdapter;
    SIZE_T maxVram = 0;
    bool foundDiscrete = false;

    // Loop through every physical adapter found on the system
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(adapter1->GetDesc1(&desc))) {
            // Filter out software emulators
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            // Preference Logic: Select via highest Local Dedicated Video Memory
            // Integrated chips generally present small numbers (e.g. 128MB or 0MB shared)
            bool isDiscrete = (desc.DedicatedVideoMemory > 512 * 1024 * 1024);

            if (isDiscrete && !foundDiscrete) {
                // First dedicated device discovered takes overriding priority
                bestAdapter = adapter1;
                maxVram = desc.DedicatedVideoMemory;
                foundDiscrete = true;
            }
            else if (isDiscrete && foundDiscrete) {
                if (desc.DedicatedVideoMemory > maxVram) {
                    bestAdapter = adapter1;
                    maxVram = desc.DedicatedVideoMemory;
                }
            }
            else if (!foundDiscrete) {
                // If no dedicated adapter has been caught yet, accumulate best fallback integrated processor
                if (desc.DedicatedVideoMemory > maxVram || !bestAdapter) {
                    bestAdapter = adapter1;
                    maxVram = desc.DedicatedVideoMemory;
                }
            }
        }
    }

    if (bestAdapter) {
        bestAdapter.As(&g_selectedAdapter);
        DXGI_ADAPTER_DESC1 finalDesc;
        if (SUCCEEDED(g_selectedAdapter->GetDesc1(&finalDesc))) {
            g_hw.gpuDeviceName = std::wstring(finalDesc.Description);
            wchar_t hexCode[64];
            swprintf_s(hexCode, L"ID: 0x%04X, Rev: 0x%02X", finalDesc.DeviceId, finalDesc.Revision);
            g_hw.gpuDriverCode = hexCode;

            if (finalDesc.DedicatedVideoMemory > 512 * 1024 * 1024) {
                g_hw.gpuArchitectureType = L"Discrete / Dedicated Accelerator (Preferred)";
            }
            else {
                g_hw.gpuArchitectureType = L"Unified / Integrated System iGPU";
            }
        }
    }
    else {
        g_hw.gpuDeviceName = L"Standard Graphics Adapter Context";
        g_hw.gpuArchitectureType = L"Hardware Acceleration Profile Layer";
    }
}

static void QueryPowerMetrics() {
    SYSTEM_POWER_STATUS sps;
    if (GetSystemPowerStatus(&sps)) {
        // Flag state 128 indicates no system battery profile available
        if (sps.BatteryFlag == 128 || sps.BatteryLifePercent == 255) {
            g_hw.hasBattery = false;
        }
        else {
            g_hw.hasBattery = true;
            g_hw.batteryPercent = sps.BatteryLifePercent;
            g_hw.batteryCharging = (sps.ACLineStatus == 1);
            g_hw.batterySecondsLeft = sps.BatteryLifeTime;
        }
    }
}

static void QueryThermalMetrics() {
    // NOTE: There is no ring-0/WMI/vendor-SDK sensor read here. Real core
    // temperatures and voltages require a kernel driver or vendor API
    // (e.g. LibreHardwareMonitor, HWiNFO SDK) that this screensaver does not
    // link against. These values are a plausible-looking simulation derived
    // from load, NOT real sensor telemetry - the overlay label says
    // "(simulated)" so it isn't mistaken for genuine hardware data.
    float baseCpu = 44.0f + (static_cast<float>(FastRandBounded(60)) / 10.0f);
    float baseGpu = 51.0f + (static_cast<float>(FastRandBounded(40)) / 10.0f);

    if (g_hw.systemCpuPercent > 60.0) baseCpu += 12.0f;
    if (g_hw.systemGpuPercent > 60.0) baseGpu += 9.0f;

    g_hw.cpuTemperature = baseCpu;
    g_hw.gpuTemperature = baseGpu;
    g_hw.cpuCoreVoltage = 1.15f + (static_cast<float>(g_hw.systemCpuPercent) * 0.002f);
    g_hw.gpuCoreVoltage = 0.92f + (static_cast<float>(g_hw.systemGpuPercent) * 0.0015f);
}

static void InitBenchmarking() {
    if (!g_benchmarkMode) return;

    SelectPreferredGraphicsDevice();

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    g_numCores = static_cast<int>(si.dwNumberOfProcessors);
    if (g_numCores < 1) g_numCores = 1;

    // Read initial process context tick metrics
    FILETIME ftCreate, ftExit, ftKernel, ftUser;
    if (GetProcessTimes(GetCurrentProcess(), &ftCreate, &ftExit, &ftKernel, &ftUser)) {
        ULARGE_INTEGER k, u;
        k.LowPart = ftKernel.dwLowDateTime;  k.HighPart = ftKernel.dwHighDateTime;
        u.LowPart = ftUser.dwLowDateTime;    u.HighPart = ftUser.dwHighDateTime;
        g_lastKernel100ns = k.QuadPart;
        g_lastUser100ns = u.QuadPart;
    }

    // Read initial device-wide systemic metric tick matrices
    FILETIME sysIdle, sysKernel, sysUser;
    if (GetSystemTimes(&sysIdle, &sysKernel, &sysUser)) {
        ULARGE_INTEGER i, k, u;
        i.LowPart = sysIdle.dwLowDateTime;   i.HighPart = sysIdle.dwHighDateTime;
        k.LowPart = sysKernel.dwLowDateTime; k.HighPart = sysKernel.dwHighDateTime;
        u.LowPart = sysUser.dwLowDateTime;   u.HighPart = sysUser.dwHighDateTime;
        g_lastSysIdle100ns = i.QuadPart;
        g_lastSysKernel100ns = k.QuadPart;
        g_lastSysUser100ns = u.QuadPart;
    }

    g_lastCpuCheckTick = GetTickCount64();

    // Hook systemic GPU core metrics query engine
    if (PdhOpenQueryW(nullptr, 0, &g_pdhQuery) == ERROR_SUCCESS) {
        // Monitors broad graphics hardware utilization via WDDM system engine queues
        if (PdhAddEnglishCounterW(g_pdhQuery, L"\\GPU Engine(*)\\Utilization Percentage", 0, &g_pdhGpuCounter) == ERROR_SUCCESS) {
            PdhCollectQueryData(g_pdhQuery);
            g_pdhAvailable = true;
        }
    }

    g_fpsWindowStart = GetTickCount64();
    g_fpsFrameCount = 0;
}

static void ShutdownBenchmarking() {
    if (g_pdhQuery) {
        PdhCloseQuery(g_pdhQuery);
        g_pdhQuery = nullptr;
    }
}

static void UpdateHardwareDiagnostics() {
    if (!g_benchmarkMode) return;

    ++g_fpsFrameCount;
    ULONGLONG now = GetTickCount64();
    ULONGLONG elapsed = now - g_fpsWindowStart;

    if (elapsed >= 1000) {
        g_hw.systemFps = g_fpsFrameCount * 1000.0 / static_cast<double>(elapsed);
        g_fpsFrameCount = 0;
        g_fpsWindowStart = now;

        g_hw.totalFpsHistory.Push(static_cast<float>(g_hw.systemFps));

        // --- Process (Internal Screen Saver Program) Profiles ---
        FILETIME ftCreate, ftExit, ftKernel, ftUser;
        if (GetProcessTimes(GetCurrentProcess(), &ftCreate, &ftExit, &ftKernel, &ftUser)) {
            ULARGE_INTEGER k, u;
            k.LowPart = ftKernel.dwLowDateTime;  k.HighPart = ftKernel.dwHighDateTime;
            u.LowPart = ftUser.dwLowDateTime;    u.HighPart = ftUser.dwHighDateTime;

            ULONGLONG kernelDelta = k.QuadPart - g_lastKernel100ns;
            ULONGLONG userDelta = u.QuadPart - g_lastUser100ns;
            ULONGLONG totalProcTime = kernelDelta + userDelta;
            ULONGLONG wallDelta = (now - g_lastCpuCheckTick) * 10000ULL;

            if (wallDelta > 0) {
                g_hw.programCpuPercent = (100.0 * static_cast<double>(totalProcTime)) / (static_cast<double>(wallDelta) * g_numCores);
            }
            g_lastKernel100ns = k.QuadPart;
            g_lastUser100ns = u.QuadPart;
        }

        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
            g_hw.programRamUsedMB = static_cast<double>(pmc.PrivateUsage) / (1024.0 * 1024.0);
        }

        if (g_selectedAdapter) {
            DXGI_QUERY_VIDEO_MEMORY_INFO vramInfo;
            // Target segment group index 0 (Local VRAM)
            if (SUCCEEDED(g_selectedAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vramInfo))) {
                g_hw.programVramUsedMB = static_cast<double>(vramInfo.CurrentUsage) / (1024.0 * 1024.0);
            }
        }

        // --- Global Device-wide Telemetry Profiles ---
        FILETIME sysIdle, sysKernel, sysUser;
        if (GetSystemTimes(&sysIdle, &sysKernel, &sysUser)) {
            ULARGE_INTEGER i, k, u;
            i.LowPart = sysIdle.dwLowDateTime;   i.HighPart = sysIdle.dwHighDateTime;
            k.LowPart = sysKernel.dwLowDateTime; k.HighPart = sysKernel.dwHighDateTime;
            u.LowPart = sysUser.dwLowDateTime;   u.HighPart = sysUser.dwHighDateTime;

            ULONGLONG idleD = i.QuadPart - g_lastSysIdle100ns;
            ULONGLONG kernD = k.QuadPart - g_lastSysKernel100ns;
            ULONGLONG userD = u.QuadPart - g_lastSysUser100ns;
            ULONGLONG sysTotal = kernD + userD;

            if (sysTotal > 0) {
                g_hw.systemCpuPercent = (100.0 * static_cast<double>(sysTotal - idleD)) / static_cast<double>(sysTotal);
            }
            g_lastSysIdle100ns = i.QuadPart;
            g_lastSysKernel100ns = k.QuadPart;
            g_lastSysUser100ns = u.QuadPart;
        }
        g_hw.totalCpuHistory.Push(static_cast<float>(g_hw.systemCpuPercent));

        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            g_hw.systemRamTotalGB = static_cast<double>(memInfo.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
            g_hw.systemRamUsedGB = static_cast<double>(memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
            g_hw.systemRamPercent = static_cast<int>(memInfo.dwMemoryLoad);
        }
        g_hw.totalRamHistory.Push(static_cast<float>(g_hw.systemRamPercent));

        if (g_pdhAvailable) {
            PdhCollectQueryData(g_pdhQuery);
            DWORD size = 0, count = 0;
            PdhGetFormattedCounterArrayW(g_pdhGpuCounter, PDH_FMT_DOUBLE, &size, &count, nullptr);
            if (size > 0) {
                std::vector<char> buf(size);
                auto items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
                if (PdhGetFormattedCounterArrayW(g_pdhGpuCounter, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
                    double highestUsage = 0.0;
                    for (DWORD idx = 0; idx < count; ++idx) {
                        if ((items[idx].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA ||
                            items[idx].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) &&
                            items[idx].FmtValue.doubleValue > highestUsage) {
                            highestUsage = items[idx].FmtValue.doubleValue;
                        }
                    }
                    g_hw.systemGpuPercent = highestUsage;
                }
            }
        }
        // Normalize out background rendering tracking drops
        if (g_hw.systemGpuPercent > 100.0) g_hw.systemGpuPercent = 100.0;
        g_hw.totalGpuHistory.Push(static_cast<float>(g_hw.systemGpuPercent));

        g_lastCpuCheckTick = now;

        QueryPowerMetrics();
        QueryThermalMetrics();
    }
}

static void InitColumns(int width, int height) {
    g_columns.clear();
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerConfig& cfg = LAYER_CONFIGS[layer];
        int colWidth = cfg.fontSize;
        int maxCols = (width - cfg.columnOffsetPx) / colWidth;
        if (maxCols < 1) maxCols = 1;

        for (int c = 0; c < maxCols; ++c) {
            if (FastRandFloat01() > cfg.columnDensity) continue;

            ColumnState col;
            col.x = cfg.columnOffsetPx + c * colWidth;
            col.layer = layer;
            col.speed = 4 + static_cast<int>(FastRandBounded(6));
            col.length = 4 + static_cast<int>(FastRandBounded(36));
            int startY = (height > 0) ? -static_cast<int>(FastRandBounded(static_cast<uint32_t>(height))) : 0;

            col.symbols.reserve(col.length);
            for (int i = 0; i < col.length; ++i) {
                Symbol s;
                s.y = static_cast<float>(startY - i * cfg.fontSize);
                s.speed = col.speed;
                s.value = RandomKatakana();
                s.interval = 5 + static_cast<int>(FastRandBounded(25));
                s.isHead = (i == 0);
                col.symbols.push_back(s);
            }
            g_columns.push_back(std::move(col));
        }
    }

    // Rebuild the per-layer index buckets once, right after g_columns is
    // finalized, so DrawFrame never has to filter col.layer on every frame.
    g_columnsByLayer.assign(NUM_LAYERS, {});
    for (size_t i = 0; i < g_columns.size(); ++i) {
        g_columnsByLayer[g_columns[i].layer].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// Direct2D Setup Execution Contracts
// ---------------------------------------------------------------------------
static HRESULT CreateDeviceResources(HWND hwnd) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

    HRESULT hr = g_d2dFactory->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(
            D2D1_RENDER_TARGET_TYPE_DEFAULT,
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE),
            0.0f, 0.0f, D2D1_RENDER_TARGET_USAGE_NONE),
        D2D1::HwndRenderTargetProperties(hwnd, size, D2D1_PRESENT_OPTIONS_IMMEDIATELY),
        &g_renderTarget);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 0.16f), &g_fadeBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 1.0f, 0.25f, 1.0f), &g_overlayBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.95f, 0.4f, 0.9f), &g_sparklineStrokeBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.35f, 0.08f, 0.22f), &g_sparklineFillBrush);
    if (FAILED(hr)) return hr;

    // Cached once instead of being recreated every frame inside
    // DrawBenchmarkOverlay (previously 2 CreateSolidColorBrush calls x ~60/sec).
    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.01f, 0.04f, 0.01f, 0.85f), &g_overlayBgBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.7f, 0.2f, 0.5f), &g_overlayBorderBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateCompatibleRenderTarget(&g_trailTarget);
    if (FAILED(hr)) return hr;

    g_trailTarget->BeginDraw();
    g_trailTarget->Clear(D2D1::ColorF(D2D1::ColorF::Black));
    g_trailTarget->EndDraw();

    return S_OK;
}

static HRESULT CreateOverlayTextFormat() {
    HRESULT hr = g_dwriteFactory->CreateTextFormat(
        L"Consolas", nullptr,
        DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
        13.0f, L"", &g_overlayTextFormat);
    if (FAILED(hr)) return hr;

    g_overlayTextFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    g_overlayTextFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
    g_overlayTextFormat->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    return S_OK;
}

static HRESULT BuildOneAtlasBitmap(IDWriteTextFormat* format, int cell, D2D1_COLOR_F color, ComPtr<ID2D1Bitmap>& outBitmap) {
    const int atlasWidth = cell * KATAKANA_COUNT;
    D2D1_SIZE_F atlasSize = D2D1::SizeF(static_cast<float>(atlasWidth), static_cast<float>(cell));

    ComPtr<ID2D1BitmapRenderTarget> buildTarget;
    HRESULT hr = g_renderTarget->CreateCompatibleRenderTarget(atlasSize, &buildTarget);
    if (FAILED(hr)) return hr;

    ComPtr<ID2D1SolidColorBrush> brush;
    buildTarget->CreateSolidColorBrush(color, &brush);

    buildTarget->BeginDraw();
    buildTarget->Clear(D2D1::ColorF(0, 0, 0, 0));
    for (int i = 0; i < KATAKANA_COUNT; ++i) {
        wchar_t ch = static_cast<wchar_t>(0x30A0 + i);
        D2D1_RECT_F rect = D2D1::RectF(
            static_cast<float>(i * cell), 0.0f,
            static_cast<float>(i * cell + cell), static_cast<float>(cell));
        buildTarget->DrawTextW(&ch, 1, format, rect, brush.Get());
    }
    hr = buildTarget->EndDraw();
    if (FAILED(hr)) return hr;

    return buildTarget->GetBitmap(&outBitmap);
}

static HRESULT BuildAtlasForLayer(int layer) {
    const LayerConfig& cfg = LAYER_CONFIGS[layer];
    const int cell = cfg.fontSize + 4;

    ComPtr<IDWriteTextFormat> format;
    HRESULT hr = g_dwriteFactory->CreateTextFormat(
        L"MS Mincho", nullptr,
        DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
        static_cast<float>(cfg.fontSize), L"", &format);
    if (FAILED(hr)) return hr;

    hr = BuildOneAtlasBitmap(format.Get(), cell,
        D2D1::ColorF(cfg.headR / 255.0f, cfg.headG / 255.0f, cfg.headB / 255.0f, 1.0f),
        g_atlases[layer].headBitmap);
    if (FAILED(hr)) return hr;

    hr = BuildOneAtlasBitmap(format.Get(), cell,
        D2D1::ColorF(cfg.trailR / 255.0f, cfg.trailG / 255.0f, cfg.trailB / 255.0f, 1.0f),
        g_atlases[layer].trailBitmap);
    if (FAILED(hr)) return hr;

    g_atlases[layer].cellWidth = cell;
    g_atlases[layer].cellHeight = cell;
    g_atlases[layer].glyphCount = KATAKANA_COUNT;
    return S_OK;
}

static void DiscardDeviceResources() {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        g_atlases[layer].headBitmap.Reset();
        g_atlases[layer].trailBitmap.Reset();
    }
    g_trailTarget.Reset();
    g_overlayBorderBrush.Reset();
    g_overlayBgBrush.Reset();
    g_sparklineStrokeBrush.Reset();
    g_sparklineFillBrush.Reset();
    g_overlayBrush.Reset();
    g_fadeBrush.Reset();
    g_renderTarget.Reset();
}

static HRESULT InitDirect2D(HWND hwnd) {
    HRESULT hr = D2D1CreateFactory<ID2D1Factory>(D2D1_FACTORY_TYPE_SINGLE_THREADED, g_d2dFactory.GetAddressOf());
    if (FAILED(hr)) return hr;

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(g_dwriteFactory.GetAddressOf()));
    if (FAILED(hr)) return hr;

    hr = CreateDeviceResources(hwnd);
    if (FAILED(hr)) return hr;

    hr = CreateOverlayTextFormat();
    if (FAILED(hr)) return hr;

    for (int i = 0; i < NUM_LAYERS; ++i) {
        hr = BuildAtlasForLayer(i);
        if (FAILED(hr)) return hr;
    }
    return S_OK;
}

// ---------------------------------------------------------------------------
// Hardware Visualization Overlay Engine
// ---------------------------------------------------------------------------
static void DrawDeviceSparkline(ID2D1RenderTarget* rt, const PerformanceHistory& history, D2D1_RECT_F bounds, float maxVal) {
    ComPtr<ID2D1PathGeometry> pathGeom;
    if (FAILED(g_d2dFactory->CreatePathGeometry(&pathGeom))) return;

    ComPtr<ID2D1GeometrySink> sink;
    if (FAILED(pathGeom->Open(&sink))) return;

    float stepX = (bounds.right - bounds.left) / static_cast<float>(SPARKLINE_HISTORY_SIZE - 1);
    float h = bounds.bottom - bounds.top;

    auto CalculateMapNode = [&](int i) -> D2D1_POINT_2F {
        float v = history.GetAt(i);
        if (v > maxVal) v = maxVal;
        if (v < 0.0f)   v = 0.0f;
        return D2D1::Point2F(bounds.left + (i * stepX), bounds.top + ((1.0f - (v / maxVal)) * h));
    };

    // Compute every mapped point exactly once and reuse it for both the
    // filled area and the stroked line, instead of calling
    // CalculateMapNode() twice per sample.
    D2D1_POINT_2F points[SPARKLINE_HISTORY_SIZE];
    for (int i = 0; i < SPARKLINE_HISTORY_SIZE; ++i) {
        points[i] = CalculateMapNode(i);
    }

    sink->BeginFigure(D2D1::Point2F(bounds.left, bounds.bottom), D2D1_FIGURE_BEGIN_FILLED);
    for (int i = 0; i < SPARKLINE_HISTORY_SIZE; ++i) {
        sink->AddLine(points[i]);
    }
    sink->AddLine(D2D1::Point2F(bounds.right, bounds.bottom));
    sink->EndFigure(D2D1_FIGURE_END_CLOSED);
    sink->Close();

    rt->FillGeometry(pathGeom.Get(), g_sparklineFillBrush.Get());

    ComPtr<ID2D1PathGeometry> lineGeom;
    if (SUCCEEDED(g_d2dFactory->CreatePathGeometry(&lineGeom))) {
        ComPtr<ID2D1GeometrySink> lineSink;
        if (SUCCEEDED(lineGeom->Open(&lineSink))) {
            lineSink->BeginFigure(points[0], D2D1_FIGURE_BEGIN_HOLLOW);
            for (int i = 1; i < SPARKLINE_HISTORY_SIZE; ++i) {
                lineSink->AddLine(points[i]);
            }
            lineSink->EndFigure(D2D1_FIGURE_END_OPEN);
            lineSink->Close();
            rt->DrawGeometry(lineGeom.Get(), g_sparklineStrokeBrush.Get(), 1.5f);
        }
    }
}

static void DrawBenchmarkOverlay(ID2D1RenderTarget* rt) {
    if (!g_benchmarkMode) return;

    std::wstringstream ss;
    ss << L"================== SOFTWARE TELEMETRY PROCESS MATRIX ==================\n";
    ss << L"PROGRAM CPU LOAD:   " << std::fixed << std::setprecision(2) << g_hw.programCpuPercent << L" %\n";
    ss << L"PROGRAM RAM CORES:  " << std::fixed << std::setprecision(1) << g_hw.programRamUsedMB << L" MB Working Set\n";
    ss << L"PROGRAM LOCAL VRAM: " << std::fixed << std::setprecision(1) << g_hw.programVramUsedMB << L" MB Allocated Segment\n\n";

    ss << L"================== DEVICE HARDWARE MONITOR PIPELINES ==================\n";
    ss << L"HARDWARE ADAPTER:   " << g_hw.gpuDeviceName << L"\n";
    ss << L"CLASSIFICATION:     " << g_hw.gpuArchitectureType << L"\n";
    ss << L"DEVICE HARDWARE ID: " << g_hw.gpuDriverCode << L"\n\n";

    ss << L"TOTAL DEVICE FPS:   " << std::fixed << std::setprecision(1) << g_hw.systemFps << L" Frame Cadence\n\n";
    ss << L"TOTAL DEVICE CPU:   " << std::fixed << std::setprecision(1) << g_hw.systemCpuPercent << L" %  [Core Vtg (sim): " << g_hw.cpuCoreVoltage << L" V]\n\n";
    ss << L"TOTAL DEVICE GPU:   " << std::fixed << std::setprecision(1) << g_hw.systemGpuPercent << L" %  [Core Vtg (sim): " << g_hw.gpuCoreVoltage << L" V]\n\n";
    ss << L"TOTAL DEVICE RAM:   " << g_hw.systemRamPercent << L" %  [" << g_hw.systemRamUsedGB << L" GB Used / " << g_hw.systemRamTotalGB << L" GB Total]\n\n";

    ss << L"CORE TEMP (SIMULATED): CPU Core Thermal: " << std::fixed << std::setprecision(1) << g_hw.cpuTemperature << L" \u00B0C | GPU Hotspot: " << g_hw.gpuTemperature << L" \u00B0C\n";

    if (g_hw.hasBattery) {
        ss << L"POWER CAPACITOR:    System Battery: " << g_hw.batteryPercent << L" % "
            << (g_hw.batteryCharging ? L"[Charging Operations]" : L"[Discharging Mode]");
    }
    else {
        ss << L"POWER CAPACITOR:    AC Wall Circuit Connected (No Battery Mod)";
    }

    std::wstring outStr = ss.str();

    // Sizing allocations for full telemetry monitoring dashboard panel interface layout
    D2D1_RECT_F layoutRect = D2D1::RectF(20.0f, 20.0f, 760.0f, 410.0f);

    rt->FillRectangle(layoutRect, g_overlayBgBrush.Get());
    rt->DrawRectangle(layoutRect, g_overlayBorderBrush.Get(), 1.0f);

    rt->DrawTextW(outStr.c_str(), static_cast<UINT32>(outStr.length()), g_overlayTextFormat.Get(),
        D2D1::RectF(layoutRect.left + 15, layoutRect.top + 15, layoutRect.right - 15, layoutRect.bottom - 15),
        g_overlayBrush.Get());

    // --- System Level Graph Node Alignments ---
    float graphX = layoutRect.right - 210.0f;
    float graphW = 195.0f;
    float graphH = 26.0f;

    // Line 11: Total Device FPS Alignment mapping
    D2D1_RECT_F graphFpsRect = D2D1::RectF(graphX, layoutRect.top + 158.0f, graphX + graphW, layoutRect.top + 158.0f + graphH);
    DrawDeviceSparkline(rt, g_hw.totalFpsHistory, graphFpsRect, 120.0f);

    // Line 13: Total Device CPU Alignment mapping
    D2D1_RECT_F graphCpuRect = D2D1::RectF(graphX, layoutRect.top + 196.0f, graphX + graphW, layoutRect.top + 196.0f + graphH);
    DrawDeviceSparkline(rt, g_hw.totalCpuHistory, graphCpuRect, 100.0f);

    // Line 15: Total Device GPU Alignment mapping
    D2D1_RECT_F graphGpuRect = D2D1::RectF(graphX, layoutRect.top + 234.0f, graphX + graphW, layoutRect.top + 234.0f + graphH);
    DrawDeviceSparkline(rt, g_hw.totalGpuHistory, graphGpuRect, 100.0f);

    // Line 17: Total Device RAM Alignment mapping
    D2D1_RECT_F graphRamRect = D2D1::RectF(graphX, layoutRect.top + 272.0f, graphX + graphW, layoutRect.top + 272.0f + graphH);
    DrawDeviceSparkline(rt, g_hw.totalRamHistory, graphRamRect, 100.0f);
}

static void DrawFrame(DWORD tickCount) {
    if (!g_renderTarget || !g_trailTarget) return;

    g_trailTarget->BeginDraw();
    g_trailTarget->FillRectangle(D2D1::RectF(0, 0, static_cast<float>(g_width), static_cast<float>(g_height)), g_fadeBrush.Get());

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerConfig& cfg = LAYER_CONFIGS[layer];
        const GlyphAtlas& atlas = g_atlases[layer];
        const float cellF = static_cast<float>(atlas.cellWidth);

        for (size_t colIdx : g_columnsByLayer[layer]) {
            ColumnState& col = g_columns[colIdx];

            for (auto& s : col.symbols) {
                if (s.interval > 0 && (tickCount % s.interval) == 0) {
                    s.value = RandomKatakana();
                }

                s.y += static_cast<float>(s.speed) * cfg.speedMul * 0.6f;
                if (s.y > g_height) s.y = -static_cast<float>(cfg.fontSize);

                int glyphIndex = static_cast<int>(s.value) - 0x30A0;
                if (glyphIndex < 0 || glyphIndex >= atlas.glyphCount) continue;

                D2D1_RECT_F srcRect = D2D1::RectF(glyphIndex * cellF, 0.0f, glyphIndex * cellF + cellF, cellF);
                D2D1_RECT_F destRect = D2D1::RectF(static_cast<float>(col.x), s.y, static_cast<float>(col.x) + cellF, s.y + cellF);

                ID2D1Bitmap* bmp = s.isHead ? atlas.headBitmap.Get() : atlas.trailBitmap.Get();
                g_trailTarget->DrawBitmap(bmp, destRect, cfg.brightnessMul, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, srcRect);
            }
        }
    }
    g_trailTarget->EndDraw();

    ComPtr<ID2D1Bitmap> trailBitmap;
    g_trailTarget->GetBitmap(&trailBitmap);

    g_renderTarget->BeginDraw();
    g_renderTarget->DrawBitmap(trailBitmap.Get(), D2D1::RectF(0, 0, static_cast<float>(g_width), static_cast<float>(g_height)));
    DrawBenchmarkOverlay(g_renderTarget.Get());
    HRESULT hr = g_renderTarget->EndDraw();

    if (hr == D2DERR_RECREATE_TARGET) {
        DiscardDeviceResources();
        if (SUCCEEDED(CreateDeviceResources(g_hwnd))) {
            for (int i = 0; i < NUM_LAYERS; ++i) BuildAtlasForLayer(i);
        }
    }

    UpdateHardwareDiagnostics();
}

static void ResetMouseTracking(HWND hwnd) {
    GetCursorPos(&g_lastMousePos);
    g_mouseInit = true;
}

// ---------------------------------------------------------------------------
// Messaging System Window Pipelines
// ---------------------------------------------------------------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        RECT rc;
        GetClientRect(hwnd, &rc);
        g_width = rc.right - rc.left;
        g_height = rc.bottom - rc.top;

        SeedFastRng(static_cast<uint32_t>(time(nullptr)) ^ static_cast<uint32_t>(GetTickCount64()));
        g_startTick = GetTickCount64();

        if (FAILED(InitDirect2D(hwnd))) {
            MessageBoxW(hwnd, L"Direct2D Core Telemetry Interface Failure.", L"Matrix Screensaver", MB_OK | MB_ICONERROR);
            DestroyWindow(hwnd);
            return 0;
        }

        InitColumns(g_width, g_height);
        InitBenchmarking();

        if (!g_isPreview) ShowCursor(FALSE);
        ResetMouseTracking(hwnd);
        SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL_MS, nullptr);
        return 0;
    }
    case WM_TIMER: {
        DrawFrame(static_cast<DWORD>(GetTickCount64()));
        return 0;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_SIZE: {
        if (g_renderTarget) {
            RECT rc;
            GetClientRect(hwnd, &rc);
            g_width = rc.right - rc.left;
            g_height = rc.bottom - rc.top;
            D2D1_SIZE_U size = D2D1::SizeU(g_width, g_height);
            g_renderTarget->Resize(size);

            // g_trailTarget (the offscreen bitmap render target holding the
            // fading trail buffer) was previously never rebuilt here, so
            // after any resize it stayed at the old dimensions while
            // g_renderTarget grew/shrank - the trail bitmap would then be
            // stretched onto the new-size render target every frame.
            // Recreate it at the new size and reseed the columns so the
            // rain layout matches the new window bounds.
            g_trailTarget.Reset();
            if (SUCCEEDED(g_renderTarget->CreateCompatibleRenderTarget(&g_trailTarget))) {
                g_trailTarget->BeginDraw();
                g_trailTarget->Clear(D2D1::ColorF(D2D1::ColorF::Black));
                g_trailTarget->EndDraw();
            }
            InitColumns(g_width, g_height);
        }
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;

    case WM_SETCURSOR:
        if (!g_isPreview) {
            SetCursor(nullptr);
            return TRUE;
        }
        break;

    case WM_MOUSEMOVE: {
        if (g_isPreview) break;
        if (InGracePeriod()) break;
        if (!g_mouseInit) { ResetMouseTracking(hwnd); break; }
        POINT p;
        GetCursorPos(&p);
        if (std::abs(p.x - g_lastMousePos.x) > MOVE_THRESHOLD || std::abs(p.y - g_lastMousePos.y) > MOVE_THRESHOLD) {
            DestroyWindow(hwnd);
        }
        return 0;
    }
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_KEYDOWN:
        if (!g_isPreview && !InGracePeriod()) DestroyWindow(hwnd);
        return 0;

    case WM_ACTIVATE:
        if (!g_isPreview && !InGracePeriod() && LOWORD(wParam) == WA_INACTIVE) DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        KillTimer(hwnd, TIMER_ID);
        if (!g_isPreview) ShowCursor(TRUE);
        ShutdownBenchmarking();
        DiscardDeviceResources();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static HWND CreateFullscreenWindow(HINSTANCE hInstance) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"MatrixScreensaverClass";
    wc.hCursor = nullptr;
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_TOPMOST, wc.lpszClassName, L"Matrix Screensaver",
        WS_POPUP | WS_VISIBLE, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, hInstance, nullptr);
    return hwnd;
}

static HWND CreatePreviewWindow(HINSTANCE hInstance, HWND parent) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"MatrixScreensaverPreviewClass";
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassW(&wc);

    RECT rc;
    GetClientRect(parent, &rc);
    return CreateWindowExW(0, wc.lpszClassName, L"", WS_CHILD | WS_VISIBLE,
        0, 0, rc.right - rc.left, rc.bottom - rc.top, parent, nullptr, hInstance, nullptr);
}

static void ShowConfigDialog(HINSTANCE hInstance, HWND ownerHwnd) {
    MessageBoxW(ownerHwnd,
        L"Matrix Hardware Diagnostics Telemetry Engine\n\nRun with the executable argument flag '/b' to activate real-time system monitoring overlays alongside the Direct2D rendering pipeline.",
        L"Settings Profile", MB_OK | MB_ICONINFORMATION);
}

// ---------------------------------------------------------------------------
// Runtime Entry Directives
// ---------------------------------------------------------------------------
int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(nCmdShow);

    std::wstring cmd(lpCmdLine ? lpCmdLine : L"");
    auto lower = cmd;
    for (auto& ch : lower) ch = towlower(ch);

    bool isConfig = false, isPreview = false;
    HWND targetParent = nullptr;

    if (lower.find(L"/b") != std::wstring::npos || lower.find(L"-b") != std::wstring::npos) g_benchmarkMode = true;

    if (lower.find(L"/c") != std::wstring::npos || lower.find(L"-c") != std::wstring::npos) {
        isConfig = true;
        size_t colon = lower.find(L':');
        if (colon != std::wstring::npos) {
            targetParent = reinterpret_cast<HWND>(static_cast<INT_PTR>(_wtoi(cmd.c_str() + colon + 1)));
        }
    }
    else if (lower.find(L"/p") != std::wstring::npos || lower.find(L"-p") != std::wstring::npos) {
        isPreview = true;
        size_t pos = lower.find(L"/p");
        if (pos == std::wstring::npos) pos = lower.find(L"-p");
        size_t numStart = cmd.find_first_of(L"0123456789", pos);
        if (numStart != std::wstring::npos) {
            targetParent = reinterpret_cast<HWND>(static_cast<INT_PTR>(wcstoll(cmd.c_str() + numStart, nullptr, 10)));
        }
    }

    if (isConfig) {
        ShowConfigDialog(hInstance, targetParent);
        return 0;
    }

    if (isPreview && targetParent && IsWindow(targetParent)) {
        g_isPreview = true;
        g_hwnd = CreatePreviewWindow(hInstance, targetParent);
    }
    else {
        g_isPreview = false;
        g_hwnd = CreateFullscreenWindow(hInstance);
    }

    if (!g_hwnd) return 0;

    ShowWindow(g_hwnd, SW_SHOW);
    UpdateWindow(g_hwnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return static_cast<int>(msg.wParam);
}