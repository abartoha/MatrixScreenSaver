// Source.cpp
// A GPU-accelerated Windows screensaver (.scr) implementing a falling-katakana
// "Matrix rain" effect using Direct3D11 instanced rendering, with multiple
// parallax depth layers. Direct2D + DirectWrite are used only once at
// startup/resize to rasterize the glyph atlases (via D3D11/D2D interop);
// all per-frame drawing is done through D3D11 instancing so each layer's
// entire visible glyph set is issued as a single draw call.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <vector>
#include <memory>
#include <string>
#include <deque>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <windows.h>
#include <shellscalingapi.h>
#include <psapi.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <d2d1.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "Psapi.lib")
#pragma comment(lib, "Pdh.lib")

// ---------------------------------------------------------------------------
// Build-time switches
// ---------------------------------------------------------------------------
// Set to 0 to fully compile out the benchmark/diagnostics overlay and all its
// bookkeeping (PDH queries, DXGI memory queries, frame-time history, HUD
// drawing). Set to 1 to include it (still toggleable at runtime with the 'B'
// key when enabled here).
#define ENABLE_BENCHMARK_OVERLAY 1

// Set to 1 to disable vsync (Present(0,0)) and let the render loop run as fast
// as the GPU can produce frames, uncapped by the monitor's refresh rate. This
// is mainly useful for measuring true max GPU throughput. Leave at 0 for
// normal use: Present(1,0) paces the loop to the monitor's native refresh
// rate (e.g. 144Hz on a 144Hz display), which is smooth and power-efficient.
#define UNCAP_FRAMERATE 0

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Hints to the OS/driver to prefer the high-performance (discrete) GPU on
// hybrid (Optimus/AMD Switchable Graphics) laptops. These are read by the
// NVIDIA and AMD drivers respectively when present as exports of the .exe.
// ---------------------------------------------------------------------------
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int   AmdPowerXpressRequestHighPerformance = 1;
}

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
    DWORD   nextChangeTick = 0; // precomputed tick at which value will re-roll
};

struct ColumnState {
    int x = 0;
    int layer = 0;
    int speed = 0;
    int length = 0;
    std::vector<Symbol> symbols;
};

struct GlyphInstance {
    float destX, destY;
    float destW, destH;
    float u0, v0;
    float u1, v1;
    float colorR, colorG, colorB, colorA;
};

struct GlyphAtlas {
    ComPtr<ID3D11Texture2D>          headTexture, trailTexture;
    ComPtr<ID3D11ShaderResourceView> headSRV, trailSRV;
    int   cellWidth = 0;
    int   cellHeight = 0;
    int   glyphCount = 0;
};
// Atlases are monitor-independent (same glyph textures reused by every
// window), so these remain shared/global rather than moving into
// MonitorWindow.
static GlyphAtlas g_atlases[NUM_LAYERS];

struct LayerInstanceBuffer {
    ComPtr<ID3D11Buffer> buffer;
    UINT capacity = 0;
};

// ---------------------------------------------------------------------------
// Multi-monitor architecture
// ---------------------------------------------------------------------------
// A single ID3D11Device/ID3D11DeviceContext is shared across all monitors
// (device objects, shaders, input layout, blend/sampler states, the glyph
// atlases, and the white 1x1 texture are all monitor-independent and created
// exactly once). Each physical monitor gets its own borderless fullscreen
// HWND with an *independent* IDXGISwapChain, backbuffer RTV, trail
// ping-pong texture, D2D interop bitmap (for the benchmark overlay), and its
// own column/symbol simulation state sized to that monitor's resolution.
//
// This mirrors the common "one device, many swapchains" pattern used for
// multi-head fullscreen D3D11 apps: swapchains are cheap per-output objects,
// while the device/context and anything derived purely from shader bytecode
// or static vertex data are safe and desirable to share.
struct MonitorWindow {
    HWND  hwnd = nullptr;
    HMONITOR hMonitor = nullptr;

    // Monitor geometry in virtual-desktop coordinates (can be negative --
    // e.g. a monitor placed to the left of/above the primary). This is what
    // CreateWindowExW needs, since Win32 window coordinates for
    // WS_POPUP/no-parent windows are always in virtual-desktop space.
    int x = 0, y = 0;
    int width = 0, height = 0;
    bool isPrimary = false;

    // Per-window D3D11 swapchain + render targets.
    ComPtr<IDXGISwapChain>         swapChain;
    ComPtr<ID3D11RenderTargetView> backBufferRTV;

    // Per-window trail ping-pong texture (persistent "fade" accumulation
    // buffer -- must not be shared across monitors, or motion on one screen
    // would bleed into another's trail history).
    ComPtr<ID3D11Texture2D>          trailTexture;
    ComPtr<ID3D11RenderTargetView>   trailRTV;
    ComPtr<ID3D11ShaderResourceView> trailSRV;

    // Per-window D2D interop bitmap onto this window's own backbuffer, used
    // only for the benchmark overlay.
    ComPtr<ID2D1Bitmap1> overlayD2DTarget;

    // Per-window simulation state (column layout depends on this monitor's
    // own width/height, so columns cannot be shared globally anymore).
    std::vector<ColumnState> columns;
    std::vector<std::vector<size_t>> columnsByLayer;

    // Per-window instance scratch/upload buffers -- kept separate so one
    // monitor's glyph count doesn't force a reallocation visible to another.
    std::vector<GlyphInstance> headScratch[NUM_LAYERS];
    std::vector<GlyphInstance> trailScratch[NUM_LAYERS];
    LayerInstanceBuffer headInstanceBuf[NUM_LAYERS];
    LayerInstanceBuffer trailInstanceBuf[NUM_LAYERS];
    LayerInstanceBuffer fadeBuf;
    LayerInstanceBuffer compositeBuf;

    bool isVisible = true;
};

static std::vector<std::unique_ptr<MonitorWindow>> g_monitorWindows;

// Preview mode (screensaver picker thumbnail) is unaffected by multi-monitor
// support -- it's always a single embedded child window -- so it keeps its
// own dedicated MonitorWindow-shaped state, created outside the multi-monitor
// enumeration path.
static std::unique_ptr<MonitorWindow> g_previewWindow;

// ---------------------------------------------------------------------------
// Direct3D11 Pipeline Core (shared across all monitor windows)
// ---------------------------------------------------------------------------
static ComPtr<ID3D11Device>           g_d3dDevice;
static ComPtr<ID3D11DeviceContext>    g_d3dContext;

static ComPtr<ID3D11VertexShader>  g_vertexShader;
static ComPtr<ID3D11PixelShader>   g_pixelShader;
static ComPtr<ID3D11InputLayout>   g_inputLayout;
static ComPtr<ID3D11Buffer>        g_quadVertexBuffer;
static ComPtr<ID3D11Buffer>        g_viewportCB;
static ComPtr<ID3D11SamplerState>  g_samplerState;
static ComPtr<ID3D11BlendState>    g_premulAlphaBlend;
static ComPtr<ID3D11BlendState>    g_opaqueBlend;
static ComPtr<ID3D11Texture2D>     g_whiteTexture;
static ComPtr<ID3D11ShaderResourceView> g_whiteSRV;

static ComPtr<ID2D1Factory1>      g_d2dFactory;
static ComPtr<ID2D1Device>        g_d2dDevice;
static ComPtr<ID2D1DeviceContext> g_d2dContext;
static ComPtr<IDWriteFactory>     g_dwriteFactory;

static bool g_isPreview = false;
// Tracks how many monitor/preview windows are currently alive, so WM_DESTROY
// can tell whether it's tearing down the last one (and therefore whether the
// shared device/pipeline/perf-counters should be released too) versus one of
// several still-running monitor windows.
static int  g_liveWindowCount = 0;
static ULONGLONG g_startTick = 0;
static const ULONGLONG STARTUP_GRACE_MS = 1000;
static bool InGracePeriod() { return (GetTickCount64() - g_startTick) < STARTUP_GRACE_MS; }

static POINT g_lastMousePos{};
static bool  g_mouseInit = false;

// ---------------------------------------------------------------------------
// Benchmark / Diagnostics Overlay
// ---------------------------------------------------------------------------
#if ENABLE_BENCHMARK_OVERLAY
// Toggle overlay with the "B" key (does not exit the screensaver, since normal
// key input already exits; the toggle is handled specially in WndProc before
// the exit-on-keypress logic runs).
static bool g_benchEnabled = true;

struct AdapterInfo {
    std::wstring description;
    SIZE_T       dedicatedVideoMemory = 0;
    SIZE_T       dedicatedSystemMemory = 0;
    SIZE_T       sharedSystemMemory = 0;
    UINT         vendorId = 0;
    UINT         deviceId = 0;
    bool         isChosen = false;
    bool         likelyDiscrete = false;
};

static std::vector<AdapterInfo> g_allAdapters;
static AdapterInfo              g_chosenAdapter;
static std::wstring             g_adapterSelectionNote;

// Frame timing
static LARGE_INTEGER g_qpcFrequency{};
static LARGE_INTEGER g_lastFrameQpc{};
static std::deque<double> g_frameTimesMs;      // rolling window, in ms
static const size_t       kFrameHistoryMax = 240; // ~4s at 60fps
static double g_currentFps = 0.0;
static double g_avgFrameMs = 0.0;
static double g_minFrameMs = 0.0;
static double g_maxFrameMs = 0.0;
static double g_p99FrameMs = 0.0;
static UINT64 g_totalFramesRendered = 0;
static UINT64 g_droppedPresentCount = 0; // Present() calls that returned an error

// CPU usage (process vs total system) via PDH
static PDH_HQUERY   g_pdhQuery = nullptr;
static PDH_HCOUNTER g_pdhProcessCpuCounter = nullptr;
static PDH_HCOUNTER g_pdhTotalCpuCounter = nullptr;
static double g_processCpuPercent = 0.0;
static double g_systemCpuPercent = 0.0;
static int    g_logicalCoreCount = 1;

// Memory usage
static SIZE_T g_processWorkingSetBytes = 0;
static SIZE_T g_processPrivateBytes = 0;
static DWORDLONG g_systemTotalPhysBytes = 0;
static DWORDLONG g_systemUsedPhysBytes = 0;
static double g_systemMemPercent = 0.0;

// GPU memory (best-effort, via DXGI budget query on the chosen adapter)
static SIZE_T g_gpuVideoMemUsedBytes = 0;
static SIZE_T g_gpuVideoMemBudgetBytes = 0;
static ComPtr<IDXGIAdapter3> g_dxgiAdapter3; // optional, for QueryVideoMemoryInfo

// Diagnostics refresh cadence: don't hammer PDH/DXGI budget queries every frame
static ULONGLONG g_lastStatsRefreshTick = 0;
static const ULONGLONG STATS_REFRESH_INTERVAL_MS = 500;

// D2D text resources for the overlay (created alongside other size-dependent resources).
// The overlay D2D *bitmap target* is per-window (see MonitorWindow::overlayD2DTarget)
// since it's a view onto that window's specific backbuffer surface; the text
// format and brushes here are monitor-independent and stay shared.
static ComPtr<IDWriteTextFormat>    g_overlayTextFormat;
static ComPtr<ID2D1SolidColorBrush> g_overlayTextBrush;
static ComPtr<ID2D1SolidColorBrush> g_overlayBgBrush;
#endif // ENABLE_BENCHMARK_OVERLAY

// Real-time frame pacing (always needed, independent of the benchmark overlay,
// since DrawFrame's animation now advances by measured elapsed time rather
// than a fixed per-tick assumption).
static LARGE_INTEGER g_pacingQpcFrequency{};
static LARGE_INTEGER g_pacingLastFrameQpc{};
static bool          g_pacingInitialized = false;

// ---------------------------------------------------------------------------
// Matrix Cascade Random String Generation Engine
// ---------------------------------------------------------------------------
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

static uint32_t FastRandBounded(uint32_t bound) {
    return static_cast<uint32_t>((static_cast<uint64_t>(FastRandU32()) * bound) >> 32);
}

static float FastRandFloat01() {
    return static_cast<float>(FastRandU32() >> 8) / static_cast<float>(1u << 24);
}

static wchar_t RandomKatakana() {
    return static_cast<wchar_t>(0x30A0 + FastRandBounded(KATAKANA_COUNT));
}

static const float DENSITY_BASELINE_AREA = 1920.0f * 1080.0f;

static float ComputeDensityScale(int width, int height) {
    float area = static_cast<float>(width) * static_cast<float>(height);
    if (area <= DENSITY_BASELINE_AREA) return 1.0f;
    float scale = DENSITY_BASELINE_AREA / area;
    if (scale < 0.35f) scale = 0.35f;
    return scale;
}

static void InitColumns(MonitorWindow& mw, int width, int height) {
    mw.columns.clear();
    const float densityScale = ComputeDensityScale(width, height);
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerConfig& cfg = LAYER_CONFIGS[layer];
        int colWidth = cfg.fontSize;
        int maxCols = (width - cfg.columnOffsetPx) / colWidth;
        if (maxCols < 1) maxCols = 1;

        const float effectiveDensity = cfg.columnDensity * densityScale;
        for (int c = 0; c < maxCols; ++c) {
            if (FastRandFloat01() > effectiveDensity) continue;

            ColumnState col;
            col.x = cfg.columnOffsetPx + c * colWidth;
            col.layer = layer;
            col.speed = 4 + static_cast<int>(FastRandBounded(6));
            col.length = 4 + static_cast<int>(FastRandBounded(36));

            // Seed the column's head somewhere across the *entire* fall range
            // (from fully above the screen down to fully below it), not just
            // above it. Previously every column always started above y=0 and
            // had to fall into view over time, so a freshly-created window
            // that only gets to render a handful of frames -- most notably
            // the tiny preview thumbnail in the screensaver picker, which the
            // OS only pumps for a brief moment before snapshotting it -- would
            // show a mature-looking top half (near the spawn point) and a
            // completely empty, solid-black bottom half (never reached yet).
            // Distributing the initial head position across the full height
            // (with some overflow above/below for a natural, already-falling
            // look) makes the very first rendered frame already resemble a
            // steady-state cascade, so short-lived render contexts like the
            // preview look correct immediately instead of only "warming up"
            // after several seconds of continuous animation.
            int spawnRangeTop = -(col.length * cfg.fontSize);
            int spawnRangeBottom = (height > 0) ? height : 0;
            int spawnSpan = spawnRangeBottom - spawnRangeTop;
            int startY = (spawnSpan > 0)
                ? spawnRangeTop + static_cast<int>(FastRandBounded(static_cast<uint32_t>(spawnSpan)))
                : 0;

            col.symbols.reserve(col.length);
            for (int i = 0; i < col.length; ++i) {
                Symbol s;
                s.y = static_cast<float>(startY - i * cfg.fontSize);
                s.speed = col.speed;
                s.value = RandomKatakana();
                s.interval = 400 + static_cast<int>(FastRandBounded(1400));
                s.isHead = (i == 0);
                s.nextChangeTick = static_cast<DWORD>(FastRandBounded(static_cast<uint32_t>(s.interval)));
                col.symbols.push_back(s);
            }
            mw.columns.push_back(std::move(col));
        }
    }

    mw.columnsByLayer.assign(NUM_LAYERS, {});
    for (size_t i = 0; i < mw.columns.size(); ++i) {
        mw.columnsByLayer[mw.columns[i].layer].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// Shader Source (compiled once at startup)
// ---------------------------------------------------------------------------
static const char* kShaderSource = R"(
cbuffer ViewportCB : register(b0) {
    float2 viewport;
    float2 _padVp;
};

struct VSIn {
    float2 localPos : POSITION;
    float2 destPos   : IPOS;
    float2 destSize  : ISIZE;
    float2 uv0       : IUVA;
    float2 uv1       : IUVB;
    float4 color     : ICOLOR;
};

struct VSOut {
    float4 pos   : SV_POSITION;
    float2 uv    : TEXCOORD0;
    float4 color : COLOR0;
};

VSOut VSMain(VSIn input) {
    VSOut o;
    float2 pixelPos = input.destPos + input.localPos * input.destSize;
    float2 ndc = float2(
        (pixelPos.x / viewport.x) * 2.0 - 1.0,
        1.0 - (pixelPos.y / viewport.y) * 2.0);
    o.pos = float4(ndc, 0.0, 1.0);
    o.uv = lerp(input.uv0, input.uv1, input.localPos);
    o.color = input.color;
    return o;
}

Texture2D tex : register(t0);
SamplerState samp : register(s0);

float4 PSMain(VSOut input) : SV_TARGET {
    float4 texColor = tex.Sample(samp, input.uv);
    return texColor * input.color;
}
)";

static HRESULT CompileShader(const char* entryPoint, const char* target, ComPtr<ID3DBlob>& outBlob) {
    ComPtr<ID3DBlob> errorBlob;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG;
#endif
    HRESULT hr = D3DCompile(kShaderSource, strlen(kShaderSource), nullptr, nullptr, nullptr,
        entryPoint, target, flags, 0, &outBlob, &errorBlob);
    if (FAILED(hr) && errorBlob) {
        OutputDebugStringA(static_cast<const char*>(errorBlob->GetBufferPointer()));
    }
    return hr;
}

// ---------------------------------------------------------------------------
// GPU Adapter Enumeration & Selection Reporting
// ---------------------------------------------------------------------------
#if ENABLE_BENCHMARK_OVERLAY
// Well-known PCI vendor IDs used to guess discrete vs integrated when the
// description string alone isn't conclusive.
static bool VendorIsDiscreteLikely(UINT vendorId, const std::wstring& desc) {
    // 0x10DE = NVIDIA (always discrete). 0x1002/0x1022 = AMD (mostly discrete,
    // though AMD APUs share the ID; description text disambiguates below).
    if (vendorId == 0x10DE) return true;
    std::wstring lower = desc;
    for (auto& ch : lower) ch = towlower(ch);
    if (lower.find(L"intel") != std::wstring::npos) return false;
    if (lower.find(L"microsoft basic render") != std::wstring::npos) return false;
    if (lower.find(L"radeon") != std::wstring::npos && lower.find(L" graphics") != std::wstring::npos
        && lower.find(L"rx") == std::wstring::npos) {
        // Heuristic: "AMD Radeon(TM) Graphics" (no RX model number) is typically an APU.
        return false;
    }
    if (vendorId == 0x1002 || vendorId == 0x1022) return true;
    return false;
}

static void EnumerateAndSelectAdapter(IDXGIFactory1* factory, IDXGIAdapter* chosenByD3D) {
    g_allAdapters.clear();
    g_adapterSelectionNote.clear();

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc))) {
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                AdapterInfo info;
                info.description = desc.Description;
                info.dedicatedVideoMemory = desc.DedicatedVideoMemory;
                info.dedicatedSystemMemory = desc.DedicatedSystemMemory;
                info.sharedSystemMemory = desc.SharedSystemMemory;
                info.vendorId = desc.VendorId;
                info.deviceId = desc.DeviceId;
                info.likelyDiscrete = VendorIsDiscreteLikely(desc.VendorId, info.description);
                g_allAdapters.push_back(info);
            }
        }
        adapter.Reset();
    }

    // Identify which adapter D3D actually picked (by LUID match) and figure out
    // whether a better (discrete, more VRAM) option exists that wasn't chosen.
    DXGI_ADAPTER_DESC1 chosenDesc{};
    bool haveChosenDesc = false;
    if (chosenByD3D) {
        ComPtr<IDXGIAdapter1> chosen1;
        if (SUCCEEDED(chosenByD3D->QueryInterface(IID_PPV_ARGS(&chosen1)))) {
            if (SUCCEEDED(chosen1->GetDesc1(&chosenDesc))) haveChosenDesc = true;
        }
    }

    const AdapterInfo* bestAlternative = nullptr;
    for (auto& a : g_allAdapters) {
        bool isThisTheChosenOne = haveChosenDesc &&
            (a.vendorId == chosenDesc.VendorId && a.deviceId == chosenDesc.DeviceId &&
                a.dedicatedVideoMemory == chosenDesc.DedicatedVideoMemory);
        a.isChosen = isThisTheChosenOne;
        if (isThisTheChosenOne) g_chosenAdapter = a;

        if (!isThisTheChosenOne && a.likelyDiscrete) {
            if (!bestAlternative || a.dedicatedVideoMemory > bestAlternative->dedicatedVideoMemory) {
                bestAlternative = &a;
            }
        }
    }

    if (!haveChosenDesc && !g_allAdapters.empty()) {
        g_chosenAdapter = g_allAdapters.front();
    }

    if (bestAlternative && !g_chosenAdapter.likelyDiscrete) {
        wchar_t buf[256];
        swprintf_s(buf, L"Note: running on '%s' (integrated); a discrete GPU '%s' is also present but not in use.",
            g_chosenAdapter.description.c_str(), bestAlternative->description.c_str());
        g_adapterSelectionNote = buf;
    }
    else if (g_chosenAdapter.likelyDiscrete) {
        g_adapterSelectionNote = L"Using discrete GPU (high-performance preference requested).";
    }
    else {
        g_adapterSelectionNote = L"Using integrated/only available GPU.";
    }
}

static void InitGpuMemoryQuery() {
    g_dxgiAdapter3.Reset();
    if (!g_d3dDevice) return;
    ComPtr<IDXGIDevice> dxgiDevice;
    if (FAILED(g_d3dDevice.As(&dxgiDevice))) return;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDevice->GetAdapter(&adapter))) return;
    adapter.As(&g_dxgiAdapter3); // may fail on older systems; that's fine, handled as unavailable
}

static void RefreshGpuMemoryUsage() {
    if (!g_dxgiAdapter3) { g_gpuVideoMemUsedBytes = 0; g_gpuVideoMemBudgetBytes = 0; return; }
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (SUCCEEDED(g_dxgiAdapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        g_gpuVideoMemUsedBytes = static_cast<SIZE_T>(info.CurrentUsage);
        g_gpuVideoMemBudgetBytes = static_cast<SIZE_T>(info.Budget);
    }
}

// ---------------------------------------------------------------------------
// CPU / Memory Diagnostics (PDH + PSAPI)
// ---------------------------------------------------------------------------
static void InitPerfCounters() {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    g_logicalCoreCount = static_cast<int>(si.dwNumberOfProcessors);
    if (g_logicalCoreCount < 1) g_logicalCoreCount = 1;

    if (PdhOpenQueryW(nullptr, 0, &g_pdhQuery) != ERROR_SUCCESS) {
        g_pdhQuery = nullptr;
        return;
    }

    DWORD pid = GetCurrentProcessId();
    wchar_t processCounterPath[256];
    // "% Processor Time" for the current process instance, normalized to
    // total logical cores by PDH already needs manual division; PDH reports
    // 0-100*coreCount for _Total-style counters on "Process" object across
    // all cores combined, so we divide by core count below when displaying.
    swprintf_s(processCounterPath, L"\\Process(%s)\\%% Processor Time", L"*"); // placeholder, resolved below
    // Resolve actual process instance name (can differ if multiple instances
    // of the same exe run, e.g. "MatrixScreensaver#1").
    {
        wchar_t exePath[MAX_PATH];
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::wstring exeName = exePath;
        size_t slash = exeName.find_last_of(L"\\/");
        if (slash != std::wstring::npos) exeName = exeName.substr(slash + 1);
        size_t dot = exeName.find_last_of(L'.');
        if (dot != std::wstring::npos) exeName = exeName.substr(0, dot);

        bool resolved = false;
        for (int suffix = 0; suffix < 8 && !resolved; ++suffix) {
            std::wstring instance = (suffix == 0) ? exeName : (exeName + L"#" + std::to_wstring(suffix));
            swprintf_s(processCounterPath, L"\\Process(%s)\\%% Processor Time", instance.c_str());
            PDH_HCOUNTER testCounter = nullptr;
            if (PdhAddCounterW(g_pdhQuery, processCounterPath, 0, &testCounter) == ERROR_SUCCESS) {
                // Verify this instance actually corresponds to our PID.
                wchar_t idCounterPath[300];
                swprintf_s(idCounterPath, L"\\Process(%s)\\ID Process", instance.c_str());
                PDH_HCOUNTER idCounter = nullptr;
                if (PdhAddCounterW(g_pdhQuery, idCounterPath, 0, &idCounter) == ERROR_SUCCESS) {
                    PdhCollectQueryData(g_pdhQuery);
                    PDH_FMT_COUNTERVALUE val{};
                    if (PdhGetFormattedCounterValue(idCounter, PDH_FMT_LONG, nullptr, &val) == ERROR_SUCCESS) {
                        if (static_cast<DWORD>(val.longValue) == pid) {
                            resolved = true;
                            g_pdhProcessCpuCounter = testCounter;
                        }
                    }
                    PdhRemoveCounter(idCounter);
                }
                if (!resolved) PdhRemoveCounter(testCounter);
            }
        }
    }

    PdhAddCounterW(g_pdhQuery, L"\\Processor(_Total)\\% Processor Time", 0, &g_pdhTotalCpuCounter);
    PdhCollectQueryData(g_pdhQuery); // prime the query; first formatted read needs two samples
}

static void ShutdownPerfCounters() {
    if (g_pdhQuery) {
        PdhCloseQuery(g_pdhQuery);
        g_pdhQuery = nullptr;
        g_pdhProcessCpuCounter = nullptr;
        g_pdhTotalCpuCounter = nullptr;
    }
}

static void RefreshCpuAndMemoryStats() {
    if (g_pdhQuery) {
        if (PdhCollectQueryData(g_pdhQuery) == ERROR_SUCCESS) {
            PDH_FMT_COUNTERVALUE val{};
            if (g_pdhProcessCpuCounter &&
                PdhGetFormattedCounterValue(g_pdhProcessCpuCounter, PDH_FMT_DOUBLE, nullptr, &val) == ERROR_SUCCESS) {
                // This counter sums across all cores (can exceed 100%); normalize.
                g_processCpuPercent = val.doubleValue / static_cast<double>(g_logicalCoreCount);
            }
            if (g_pdhTotalCpuCounter &&
                PdhGetFormattedCounterValue(g_pdhTotalCpuCounter, PDH_FMT_DOUBLE, nullptr, &val) == ERROR_SUCCESS) {
                g_systemCpuPercent = val.doubleValue;
            }
        }
    }

    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        g_processWorkingSetBytes = pmc.WorkingSetSize;
        g_processPrivateBytes = pmc.PrivateUsage;
    }

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        g_systemTotalPhysBytes = ms.ullTotalPhys;
        g_systemUsedPhysBytes = ms.ullTotalPhys - ms.ullAvailPhys;
        g_systemMemPercent = static_cast<double>(ms.dwMemoryLoad);
    }

    RefreshGpuMemoryUsage();
}
#else
// No-op stubs so call sites don't need scattered #ifdefs when the overlay is
// compiled out entirely.
static inline void InitPerfCounters() {}
static inline void ShutdownPerfCounters() {}
static inline void RefreshCpuAndMemoryStats() {}
#endif // ENABLE_BENCHMARK_OVERLAY

// ---------------------------------------------------------------------------
// Direct3D11 Setup Execution Contracts
// ---------------------------------------------------------------------------
// Creates the single shared ID3D11Device/ID3D11DeviceContext used by every
// monitor window. Adapter selection (explicit high-perf adapter via
// IDXGIFactory6, falling back to hardware, falling back to WARP) is
// unchanged from the original single-window version -- it just no longer
// creates a swapchain at the same time, since a device is monitor-agnostic
// but a swapchain is tied to one specific output window.
static HRESULT CreateD3DDevice() {
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL chosen{};

    UINT deviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    UINT debugFlag = 0;
#ifdef _DEBUG
    debugFlag = D3D11_CREATE_DEVICE_DEBUG;
#endif

    // Explicitly pick the high-performance adapter when the system exposes
    // IDXGIFactory6 (Windows 10 1803+). This is the most reliable way to steer
    // hybrid-graphics laptops toward the discrete GPU; the NvOptimusEnablement /
    // AmdPowerXpressRequestHighPerformance exports above are a fallback for
    // older systems where factory6 isn't available.
    ComPtr<IDXGIAdapter1> explicitAdapter;
    ComPtr<IDXGIFactory1> factory1;
    {
        ComPtr<IDXGIFactory6> factory6;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory1)))) {
            if (SUCCEEDED(factory1.As(&factory6))) {
                factory6->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    IID_PPV_ARGS(&explicitAdapter));
            }
        }
    }

    HRESULT hr;
    if (explicitAdapter) {
        hr = D3D11CreateDevice(
            explicitAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags | debugFlag,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &g_d3dDevice, &chosen, &g_d3dContext);
        if (FAILED(hr) && debugFlag != 0) {
            hr = D3D11CreateDevice(
                explicitAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags,
                levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                &g_d3dDevice, &chosen, &g_d3dContext);
        }
    }
    else {
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags | debugFlag,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &g_d3dDevice, &chosen, &g_d3dContext);

        if (FAILED(hr) && debugFlag != 0) {
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags,
                levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                &g_d3dDevice, &chosen, &g_d3dContext);
        }
    }

    if (FAILED(hr)) {
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, deviceFlags,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &g_d3dDevice, &chosen, &g_d3dContext);
    }

    if (SUCCEEDED(hr)) {
        // Report which adapter actually ended up being used, and whether a
        // better discrete option was available but skipped.
        ComPtr<IDXGIAdapter> actualAdapter;
        if (explicitAdapter) {
            actualAdapter = explicitAdapter;
        }
        else if (g_d3dDevice) {
            ComPtr<IDXGIDevice> dxgiDevice;
            if (SUCCEEDED(g_d3dDevice.As(&dxgiDevice))) {
                dxgiDevice->GetAdapter(&actualAdapter);
            }
        }
        if (!factory1) CreateDXGIFactory1(IID_PPV_ARGS(&factory1));
#if ENABLE_BENCHMARK_OVERLAY
        if (factory1) EnumerateAndSelectAdapter(factory1.Get(), actualAdapter.Get());
        InitGpuMemoryQuery();
#endif
    }

    return hr;
}

// Creates an independent swapchain for one monitor window against the
// already-created shared g_d3dDevice. Must be called once per HWND, after
// CreateD3DDevice() has succeeded. DXGI requires the factory used to create
// a swapchain be obtained from the same adapter/device chain as the device
// itself, so we pull the factory via the device's parent adapter rather than
// creating a fresh, possibly-mismatched IDXGIFactory1.
static HRESULT CreateSwapChainForWindow(HWND hwnd, int width, int height,
    ComPtr<IDXGISwapChain>& outSwapChain) {
    if (!g_d3dDevice) return E_FAIL;

    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = g_d3dDevice.As(&dxgiDevice);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIFactory1> factory;
    hr = adapter->GetParent(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return hr;

    DXGI_SWAP_CHAIN_DESC scd = {};
    scd.BufferCount = 1;
    scd.BufferDesc.Width = width;
    scd.BufferDesc.Height = height;
    scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.BufferDesc.RefreshRate.Numerator = 0;
    scd.BufferDesc.RefreshRate.Denominator = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
    scd.SampleDesc.Quality = 0;
    scd.Windowed = TRUE;
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    outSwapChain.Reset();
    hr = factory->CreateSwapChain(g_d3dDevice.Get(), &scd, &outSwapChain);
    if (FAILED(hr)) return hr;

    // Prevent DXGI's default Alt+Enter fullscreen-toggle handling from ever
    // engaging -- each window is already borderless/topmost and manually
    // sized to its own monitor's exact bounds, so DXGI-managed exclusive
    // fullscreen would fight with that per-monitor layout.
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);

    return S_OK;
}

static HRESULT CreatePipelineObjects(const wchar_t** failedCall = nullptr) {
    ComPtr<ID3DBlob> vsBlob, psBlob;
    HRESULT hr = CompileShader("VSMain", "vs_4_0", vsBlob);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CompileShader(VSMain)"; return hr; }
    hr = CompileShader("PSMain", "ps_4_0", psBlob);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CompileShader(PSMain)"; return hr; }

    hr = g_d3dDevice->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &g_vertexShader);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateVertexShader"; return hr; }
    hr = g_d3dDevice->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &g_pixelShader);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreatePixelShader"; return hr; }

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 0,  D3D11_INPUT_PER_VERTEX_DATA,   0 },
        { "IPOS",     0, DXGI_FORMAT_R32G32_FLOAT,       1, 0,  D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "ISIZE",    0, DXGI_FORMAT_R32G32_FLOAT,       1, 8,  D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "IUVA",     0, DXGI_FORMAT_R32G32_FLOAT,       1, 16, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "IUVB",     0, DXGI_FORMAT_R32G32_FLOAT,       1, 24, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "ICOLOR",   0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 32, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    };
    hr = g_d3dDevice->CreateInputLayout(layout, ARRAYSIZE(layout), vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &g_inputLayout);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateInputLayout"; return hr; }

    const float quadVerts[] = { 0,0, 1,0, 0,1, 1,1 };
    D3D11_BUFFER_DESC qbd = {};
    qbd.ByteWidth = sizeof(quadVerts);
    qbd.Usage = D3D11_USAGE_IMMUTABLE;
    qbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA qinit = { quadVerts, 0, 0 };
    hr = g_d3dDevice->CreateBuffer(&qbd, &qinit, &g_quadVertexBuffer);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBuffer(quadVertexBuffer)"; return hr; }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(float) * 4;
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = g_d3dDevice->CreateBuffer(&cbd, nullptr, &g_viewportCB);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBuffer(viewportCB)"; return hr; }

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    hr = g_d3dDevice->CreateSamplerState(&sd, &g_samplerState);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateSamplerState"; return hr; }

    D3D11_BLEND_DESC pbd = {};
    pbd.RenderTarget[0].BlendEnable = TRUE;
    pbd.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    pbd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    pbd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    pbd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    pbd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    pbd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    pbd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = g_d3dDevice->CreateBlendState(&pbd, &g_premulAlphaBlend);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBlendState(premulAlpha)"; return hr; }

    D3D11_BLEND_DESC obd = {};
    obd.RenderTarget[0].BlendEnable = FALSE;
    obd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = g_d3dDevice->CreateBlendState(&obd, &g_opaqueBlend);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBlendState(opaque)"; return hr; }

    D3D11_TEXTURE2D_DESC wtd = {};
    wtd.Width = 1; wtd.Height = 1; wtd.MipLevels = 1; wtd.ArraySize = 1;
    wtd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    wtd.SampleDesc.Count = 1;
    wtd.Usage = D3D11_USAGE_IMMUTABLE;
    wtd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const UINT32 whitePixel = 0xFFFFFFFFu;
    D3D11_SUBRESOURCE_DATA winit = { &whitePixel, sizeof(UINT32), 0 };
    hr = g_d3dDevice->CreateTexture2D(&wtd, &winit, &g_whiteTexture);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateTexture2D(whiteTexture)"; return hr; }
    hr = g_d3dDevice->CreateShaderResourceView(g_whiteTexture.Get(), nullptr, &g_whiteSRV);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateShaderResourceView(whiteSRV)"; return hr; }

    return S_OK;
}

static HRESULT CreateSizeDependentResources(MonitorWindow& mw, int width, int height) {
    ComPtr<ID3D11Texture2D> backBuffer;
    HRESULT hr = mw.swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateRenderTargetView(backBuffer.Get(), nullptr, &mw.backBufferRTV);
    if (FAILED(hr)) return hr;

    D3D11_TEXTURE2D_DESC ttd = {};
    ttd.Width = width; ttd.Height = height; ttd.MipLevels = 1; ttd.ArraySize = 1;
    ttd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    ttd.SampleDesc.Count = 1;
    ttd.Usage = D3D11_USAGE_DEFAULT;
    ttd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    hr = g_d3dDevice->CreateTexture2D(&ttd, nullptr, &mw.trailTexture);
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateRenderTargetView(mw.trailTexture.Get(), nullptr, &mw.trailRTV);
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateShaderResourceView(mw.trailTexture.Get(), nullptr, &mw.trailSRV);
    if (FAILED(hr)) return hr;

    const float black[4] = { 0, 0, 0, 1 };
    g_d3dContext->ClearRenderTargetView(mw.trailRTV.Get(), black);

#if ENABLE_BENCHMARK_OVERLAY
    // D2D bitmap view onto this window's own swap chain backbuffer surface,
    // used only to draw the benchmark overlay text on top of its composited
    // frame. Each monitor window gets its own -- it cannot be shared since
    // each backbuffer is a distinct DXGI surface.
    mw.overlayD2DTarget.Reset();
    if (g_d2dContext) {
        ComPtr<IDXGISurface> backSurface;
        if (SUCCEEDED(backBuffer.As(&backSurface))) {
            D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
                D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
            g_d2dContext->CreateBitmapFromDxgiSurface(backSurface.Get(), &bp, &mw.overlayD2DTarget);
        }
    }
#endif
    return S_OK;
}

#if ENABLE_BENCHMARK_OVERLAY
static HRESULT CreateOverlayResources() {
    if (!g_dwriteFactory || !g_d2dContext) return E_FAIL;

    HRESULT hr = g_dwriteFactory->CreateTextFormat(
        L"Consolas", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL, 14.0f, L"en-us", &g_overlayTextFormat);
    if (FAILED(hr)) return hr;
    g_overlayTextFormat->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

    hr = g_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0.35f, 1.0f, 0.45f, 1.0f), &g_overlayTextBrush);
    if (FAILED(hr)) return hr;

    hr = g_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.55f), &g_overlayBgBrush);
    return hr;
}
#else
static inline HRESULT CreateOverlayResources() { return S_OK; }
#endif // ENABLE_BENCHMARK_OVERLAY

static HRESULT BuildOneAtlasTexture(IDWriteTextFormat* format, int cell, D2D1_COLOR_F color,
    ComPtr<ID3D11Texture2D>& outTexture, ComPtr<ID3D11ShaderResourceView>& outSRV) {
    const int atlasWidth = cell * KATAKANA_COUNT;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = atlasWidth;
    desc.Height = cell;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;

    HRESULT hr = g_d3dDevice->CreateTexture2D(&desc, nullptr, &outTexture);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGISurface> surface;
    hr = outTexture.As(&surface);
    if (FAILED(hr)) return hr;

    D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED));

    ComPtr<ID2D1Bitmap1> d2dBitmap;
    hr = g_d2dContext->CreateBitmapFromDxgiSurface(surface.Get(), &bp, &d2dBitmap);
    if (FAILED(hr)) return hr;

    g_d2dContext->SetTarget(d2dBitmap.Get());

    ComPtr<ID2D1SolidColorBrush> brush;
    g_d2dContext->CreateSolidColorBrush(color, &brush);

    g_d2dContext->BeginDraw();
    g_d2dContext->Clear(D2D1::ColorF(0, 0, 0, 0));
    for (int i = 0; i < KATAKANA_COUNT; ++i) {
        wchar_t ch = static_cast<wchar_t>(0x30A0 + i);
        D2D1_RECT_F rect = D2D1::RectF(
            static_cast<float>(i * cell), 0.0f,
            static_cast<float>(i * cell + cell), static_cast<float>(cell));
        g_d2dContext->DrawTextW(&ch, 1, format, rect, brush.Get());
    }
    hr = g_d2dContext->EndDraw();
    g_d2dContext->SetTarget(nullptr);
    if (FAILED(hr)) return hr;

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    return g_d3dDevice->CreateShaderResourceView(outTexture.Get(), &srvDesc, &outSRV);
}

static HRESULT BuildAtlasForLayer(int layer) {
    const LayerConfig& cfg = LAYER_CONFIGS[layer];
    const int cell = cfg.fontSize + 4;

    ComPtr<IDWriteTextFormat> format;
    // Typo fix applied: g_dwriteFactory
    HRESULT hr = g_dwriteFactory->CreateTextFormat(
        L"MS Mincho", nullptr,
        DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
        static_cast<float>(cfg.fontSize), L"", &format);
    if (FAILED(hr)) return hr;

    hr = BuildOneAtlasTexture(format.Get(), cell,
        D2D1::ColorF(cfg.headR / 255.0f, cfg.headG / 255.0f, cfg.headB / 255.0f, 1.0f),
        g_atlases[layer].headTexture, g_atlases[layer].headSRV);
    if (FAILED(hr)) return hr;

    hr = BuildOneAtlasTexture(format.Get(), cell,
        D2D1::ColorF(cfg.trailR / 255.0f, cfg.trailG / 255.0f, cfg.trailB / 255.0f, 1.0f),
        g_atlases[layer].trailTexture, g_atlases[layer].trailSRV);
    if (FAILED(hr)) return hr;

    g_atlases[layer].cellWidth = cell;
    g_atlases[layer].cellHeight = cell;
    g_atlases[layer].glyphCount = KATAKANA_COUNT;
    return S_OK;
}

// Releases everything that is specific to one monitor window: its swapchain
// render target, trail ping-pong texture, D2D overlay bitmap, and per-window
// instance upload buffers. Does NOT touch the shared device, shaders, or
// glyph atlases -- those survive independently of any single window.
static void DiscardWindowResources(MonitorWindow& mw) {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        mw.headInstanceBuf[layer].buffer.Reset();
        mw.headInstanceBuf[layer].capacity = 0;
        mw.trailInstanceBuf[layer].buffer.Reset();
        mw.trailInstanceBuf[layer].capacity = 0;
    }
    mw.fadeBuf.buffer.Reset();
    mw.fadeBuf.capacity = 0;
    mw.compositeBuf.buffer.Reset();
    mw.compositeBuf.capacity = 0;
    mw.trailTexture.Reset();
    mw.trailRTV.Reset();
    mw.trailSRV.Reset();
    mw.backBufferRTV.Reset();
#if ENABLE_BENCHMARK_OVERLAY
    mw.overlayD2DTarget.Reset();
#endif
    mw.swapChain.Reset();
}

// Releases the shared device/context, shared pipeline objects (shaders,
// input layout, blend/sampler states, white texture), the shared glyph
// atlases, and shared D2D/DWrite objects. Callers must have already torn
// down every MonitorWindow's per-window resources (via
// DiscardWindowResources) before calling this, since those per-window
// objects were created against this device.
static void DiscardSharedDeviceResources() {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        g_atlases[layer].headTexture.Reset();
        g_atlases[layer].trailTexture.Reset();
        g_atlases[layer].headSRV.Reset();
        g_atlases[layer].trailSRV.Reset();
    }
#if ENABLE_BENCHMARK_OVERLAY
    g_overlayTextFormat.Reset();
    g_overlayTextBrush.Reset();
    g_overlayBgBrush.Reset();
    g_dxgiAdapter3.Reset();
#endif
    g_d2dContext.Reset();
    g_d2dDevice.Reset();
    g_vertexShader.Reset();
    g_pixelShader.Reset();
    g_inputLayout.Reset();
    g_quadVertexBuffer.Reset();
    g_viewportCB.Reset();
    g_samplerState.Reset();
    g_premulAlphaBlend.Reset();
    g_opaqueBlend.Reset();
    g_whiteTexture.Reset();
    g_whiteSRV.Reset();
    g_d3dContext.Reset();
    g_d3dDevice.Reset();
}

// Convenience helper for the device-lost/reset recovery path: tears down
// every monitor window's resources plus the shared device in one call.
static void DiscardAllDeviceResources() {
    for (auto& mwPtr : g_monitorWindows) {
        if (mwPtr) DiscardWindowResources(*mwPtr);
    }
    if (g_previewWindow) DiscardWindowResources(*g_previewWindow);
    DiscardSharedDeviceResources();
}

// One-time setup of everything monitor-independent: the shared D3D11
// device/context, D2D/DWrite factories, the shader/pipeline objects, the
// overlay text format/brushes, and the glyph atlases. Must succeed exactly
// once before any MonitorWindow is initialized.
static HRESULT InitSharedPipeline(const wchar_t** failedStage = nullptr) {
    HRESULT hr = CreateD3DDevice();
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateD3DDevice"; return hr; }

    hr = D2D1CreateFactory<ID2D1Factory1>(D2D1_FACTORY_TYPE_SINGLE_THREADED, g_d2dFactory.GetAddressOf());
    if (FAILED(hr)) { if (failedStage) *failedStage = L"D2D1CreateFactory"; return hr; }

    // Interop Conversion Error Fix Applied: declared as ComPtr<IDXGIDevice> instead of IDXGISurface
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = g_d3dDevice.As(&dxgiDevice);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"QueryIDXGIDevice"; return hr; }
    hr = g_d2dFactory->CreateDevice(dxgiDevice.Get(), &g_d2dDevice);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"D2D1Factory::CreateDevice"; return hr; }
    hr = g_d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &g_d2dContext);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateDeviceContext"; return hr; }

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(g_dwriteFactory.GetAddressOf()));
    if (FAILED(hr)) { if (failedStage) *failedStage = L"DWriteCreateFactory"; return hr; }

    {
        const wchar_t* pipelineCall = nullptr;
        hr = CreatePipelineObjects(&pipelineCall);
        if (FAILED(hr)) {
            if (failedStage) {
                static wchar_t detailBuf[128];
                swprintf_s(detailBuf, L"CreatePipelineObjects / %s", pipelineCall ? pipelineCall : L"(unknown)");
                *failedStage = detailBuf;
            }
            return hr;
        }
    }

    hr = CreateOverlayResources();
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateOverlayResources"; return hr; }

    QueryPerformanceFrequency(&g_pacingQpcFrequency);
    QueryPerformanceCounter(&g_pacingLastFrameQpc);
    g_pacingInitialized = true;

#if ENABLE_BENCHMARK_OVERLAY
    g_qpcFrequency = g_pacingQpcFrequency;
    g_lastFrameQpc = g_pacingLastFrameQpc;
    g_frameTimesMs.clear();
    g_totalFramesRendered = 0;
    g_droppedPresentCount = 0;
#endif

    for (int i = 0; i < NUM_LAYERS; ++i) {
        hr = BuildAtlasForLayer(i);
        if (FAILED(hr)) { if (failedStage) *failedStage = L"BuildAtlasForLayer"; return hr; }
    }
    return S_OK;
}

// Per-window setup: creates this window's swapchain and its size-dependent
// resources (backbuffer RTV, trail texture, overlay D2D bitmap) against the
// already-initialized shared device. Must be called after
// InitSharedPipeline() has succeeded at least once.
static HRESULT InitMonitorWindow(MonitorWindow& mw, const wchar_t** failedStage = nullptr) {
    RECT rc;
    GetClientRect(mw.hwnd, &rc);
    int width = rc.right - rc.left, height = rc.bottom - rc.top;
    if (width < 1) width = 1;
    if (height < 1) height = 1;

    HRESULT hr = CreateSwapChainForWindow(mw.hwnd, width, height, mw.swapChain);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateSwapChainForWindow"; return hr; }

    hr = CreateSizeDependentResources(mw, width, height);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateSizeDependentResources"; return hr; }

    return S_OK;
}

static HRESULT EnsureInstanceCapacity(LayerInstanceBuffer& lib, UINT needed) {
    if (needed <= lib.capacity && lib.buffer) return S_OK;
    UINT newCapacity = lib.capacity == 0 ? 64 : lib.capacity;
    while (newCapacity < needed) newCapacity += newCapacity / 2 + 8;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = newCapacity * sizeof(GlyphInstance);
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    lib.buffer.Reset();
    HRESULT hr = g_d3dDevice->CreateBuffer(&bd, nullptr, &lib.buffer);
    if (FAILED(hr)) { lib.capacity = 0; return hr; }
    lib.capacity = newCapacity;
    return S_OK;
}

static HRESULT UploadInstances(LayerInstanceBuffer& lib, const std::vector<GlyphInstance>& data) {
    if (data.empty()) return S_OK;
    HRESULT hr = EnsureInstanceCapacity(lib, static_cast<UINT>(data.size()));
    if (FAILED(hr)) return hr;

    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = g_d3dContext->Map(lib.buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr)) return hr;
    memcpy(mapped.pData, data.data(), data.size() * sizeof(GlyphInstance));
    g_d3dContext->Unmap(lib.buffer.Get(), 0);
    return S_OK;
}

static void DrawInstanced(ID3D11ShaderResourceView* srv, LayerInstanceBuffer& lib, UINT count) {
    if (count == 0 || !lib.buffer) return;
    ID3D11Buffer* buffers[2] = { g_quadVertexBuffer.Get(), lib.buffer.Get() };
    UINT strides[2] = { sizeof(float) * 2, sizeof(GlyphInstance) };
    UINT offsets[2] = { 0, 0 };
    g_d3dContext->IASetVertexBuffers(0, 2, buffers, strides, offsets);
    g_d3dContext->PSSetShaderResources(0, 1, &srv);
    g_d3dContext->DrawInstanced(4, count, 0, 0);
}

// ---------------------------------------------------------------------------
// Frame Rendering
// ---------------------------------------------------------------------------
#if ENABLE_BENCHMARK_OVERLAY
// Updates rolling frame-time stats (avg/min/max/p99) from g_frameTimesMs.
static void RecomputeFrameStats() {
    if (g_frameTimesMs.empty()) return;
    double sum = 0.0, mn = g_frameTimesMs[0], mx = g_frameTimesMs[0];
    for (double t : g_frameTimesMs) {
        sum += t;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
    }
    g_avgFrameMs = sum / static_cast<double>(g_frameTimesMs.size());
    g_minFrameMs = mn;
    g_maxFrameMs = mx;
    g_currentFps = (g_avgFrameMs > 0.0001) ? (1000.0 / g_avgFrameMs) : 0.0;

    std::vector<double> sorted(g_frameTimesMs.begin(), g_frameTimesMs.end());
    std::sort(sorted.begin(), sorted.end());
    size_t p99Idx = static_cast<size_t>(sorted.size() * 0.99);
    if (p99Idx >= sorted.size()) p99Idx = sorted.size() - 1;
    g_p99FrameMs = sorted[p99Idx]; // worst 1% frame time -- the classic "stutter" indicator
}

static std::wstring FormatBytesMB(SIZE_T bytes) {
    wchar_t buf[64];
    swprintf_s(buf, L"%.0f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

// Draws the benchmark/diagnostics HUD directly onto the backbuffer using D2D,
// after the D3D11 composite pass has finished writing it. This runs through
// the D3D/D2D interop device context so it composites correctly on top of
// the Matrix rain without needing its own render target.
static void DrawBenchmarkOverlay(MonitorWindow& mw) {
    if (!g_benchEnabled || !g_d2dContext || !mw.overlayD2DTarget || !g_overlayTextFormat) return;

    wchar_t buf[1024];
    int len = swprintf_s(buf,
        L"FPS: %.1f   frame: %.2f ms (min %.2f / avg %.2f / max %.2f / p99 %.2f)\n"
        L"Frames rendered: %llu   Dropped presents: %llu\n"
        L"CPU  -- this process: %.1f%%   system total: %.1f%% (%d logical cores)\n"
        L"RAM  -- this process: %s working set / %s private   system: %.0f%% used (%s / %s)\n"
        L"GPU adapter: %s%s\n"
        L"GPU VRAM: %s used / %s budget%s\n"
        L"[B] toggle this overlay",
        g_currentFps, g_avgFrameMs, g_minFrameMs, g_avgFrameMs, g_maxFrameMs, g_p99FrameMs,
        static_cast<unsigned long long>(g_totalFramesRendered),
        static_cast<unsigned long long>(g_droppedPresentCount),
        g_processCpuPercent, g_systemCpuPercent, g_logicalCoreCount,
        FormatBytesMB(g_processWorkingSetBytes).c_str(), FormatBytesMB(g_processPrivateBytes).c_str(),
        g_systemMemPercent, FormatBytesMB(g_systemUsedPhysBytes).c_str(), FormatBytesMB(g_systemTotalPhysBytes).c_str(),
        g_chosenAdapter.description.c_str(), g_chosenAdapter.likelyDiscrete ? L" (discrete)" : L" (integrated)",
        g_dxgiAdapter3 ? FormatBytesMB(g_gpuVideoMemUsedBytes).c_str() : L"n/a",
        g_dxgiAdapter3 ? FormatBytesMB(g_gpuVideoMemBudgetBytes).c_str() : L"n/a",
        g_dxgiAdapter3 ? L"" : L" (query unsupported on this driver)"
    );
    if (len < 0) return;

    g_d2dContext->SetTarget(mw.overlayD2DTarget.Get());
    g_d2dContext->BeginDraw();

    const float pad = 10.0f;
    const float panelW = 620.0f;
    const float panelH = 130.0f;
    D2D1_RECT_F bgRect = D2D1::RectF(pad, pad, pad + panelW, pad + panelH);
    g_d2dContext->FillRectangle(bgRect, g_overlayBgBrush.Get());

    D2D1_RECT_F textRect = D2D1::RectF(pad + 8.0f, pad + 6.0f, pad + panelW - 8.0f, pad + panelH - 6.0f);
    g_d2dContext->DrawTextW(buf, static_cast<UINT32>(len), g_overlayTextFormat.Get(), textRect, g_overlayTextBrush.Get());

    // Note about GPU selection (only shown when relevant, drawn as a second line
    // beneath the panel so it doesn't compete for space with the dense stats).
    if (!g_adapterSelectionNote.empty()) {
        D2D1_RECT_F noteRect = D2D1::RectF(pad, pad + panelH + 4.0f, pad + panelW, pad + panelH + 40.0f);
        g_d2dContext->DrawTextW(g_adapterSelectionNote.c_str(), static_cast<UINT32>(g_adapterSelectionNote.size()),
            g_overlayTextFormat.Get(), noteRect, g_overlayTextBrush.Get());
    }

    g_d2dContext->EndDraw();
    g_d2dContext->SetTarget(nullptr);
}
#else
static inline void DrawBenchmarkOverlay(MonitorWindow&) {}
#endif // ENABLE_BENCHMARK_OVERLAY

static void DrawFrame(MonitorWindow& mw, DWORD tickCount) {
    if (!g_d3dContext || !mw.trailRTV || !mw.backBufferRTV) return;

    // Real elapsed time since the previous frame. This always runs (regardless
    // of the benchmark macro) because animation speed must stay correct now
    // that the render loop is paced by Present()/vsync instead of a fixed
    // 16ms Win32 timer tick -- frame rate can now vary (e.g. 60 vs 144Hz), so
    // all per-frame motion must scale by real delta time rather than assuming
    // a fixed tick. Frame pacing is process-global (not per-monitor): with
    // several independent swapchains, each Present() below paces against its
    // own monitor's vblank, but the animation clock driving the simulation is
    // shared so all monitors advance in lockstep rather than drifting apart.
    LARGE_INTEGER frameStartQpc;
    QueryPerformanceCounter(&frameStartQpc);
    double deltaSeconds = 0.016; // sane fallback for the very first frame
    if (g_pacingInitialized && g_pacingQpcFrequency.QuadPart > 0) {
        deltaSeconds = static_cast<double>(frameStartQpc.QuadPart - g_pacingLastFrameQpc.QuadPart)
            / static_cast<double>(g_pacingQpcFrequency.QuadPart);
        // Clamp to avoid huge jumps after the window was minimized/stalled.
        if (deltaSeconds > 0.25) deltaSeconds = 0.25;
        if (deltaSeconds < 0.0) deltaSeconds = 0.0;
    }
    g_pacingLastFrameQpc = frameStartQpc;
    g_pacingInitialized = true;
    const float deltaTimeScale = static_cast<float>(deltaSeconds) * 60.0f; // 1.0 at 60fps, as original tuning assumed

#if ENABLE_BENCHMARK_OVERLAY
    if (g_qpcFrequency.QuadPart > 0) {
        g_frameTimesMs.push_back(deltaSeconds * 1000.0);
        while (g_frameTimesMs.size() > kFrameHistoryMax) g_frameTimesMs.pop_front();
        RecomputeFrameStats();
    }
    g_lastFrameQpc = frameStartQpc;
    ++g_totalFramesRendered;

    ULONGLONG nowTick = GetTickCount64();
    if (g_benchEnabled && (nowTick - g_lastStatsRefreshTick) >= STATS_REFRESH_INTERVAL_MS) {
        RefreshCpuAndMemoryStats();
        g_lastStatsRefreshTick = nowTick;
    }
#endif

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        mw.headScratch[layer].clear();
        mw.trailScratch[layer].clear();
    }

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerConfig& cfg = LAYER_CONFIGS[layer];
        const GlyphAtlas& atlas = g_atlases[layer];
        const float cellF = static_cast<float>(atlas.cellWidth);
        const float brightness = cfg.brightnessMul;

        for (size_t colIdx : mw.columnsByLayer[layer]) {
            ColumnState& col = mw.columns[colIdx];

            for (auto& s : col.symbols) {
                if (s.interval > 0 && tickCount >= s.nextChangeTick) {
                    s.value = RandomKatakana();
                    s.nextChangeTick = tickCount + static_cast<DWORD>(s.interval);
                }

                // Original tuning was "speed * speedMul * 0.6" per 16ms timer
                // tick (~60fps assumed). deltaTimeScale is 1.0 at 60fps and
                // scales proportionally at other frame rates, so fall speed
                // stays constant in real time regardless of how fast frames
                // are actually being produced (60Hz, 144Hz, uncapped, etc).
                s.y += static_cast<float>(s.speed) * cfg.speedMul * 0.6f * deltaTimeScale;
                if (s.y > mw.height) s.y = -static_cast<float>(cfg.fontSize);

                if (s.y + cellF < 0.0f || s.y > static_cast<float>(mw.height)) continue;

                int glyphIndex = static_cast<int>(s.value) - 0x30A0;
                if (glyphIndex < 0 || glyphIndex >= atlas.glyphCount) continue;

                const float u0 = glyphIndex / static_cast<float>(atlas.glyphCount);
                const float u1 = (glyphIndex + 1) / static_cast<float>(atlas.glyphCount);

                GlyphInstance inst;
                inst.destX = static_cast<float>(col.x);
                inst.destY = s.y;
                inst.destW = cellF;
                inst.destH = cellF;
                inst.u0 = u0; inst.v0 = 0.0f;
                inst.u1 = u1; inst.v1 = 1.0f;
                inst.colorR = inst.colorG = inst.colorB = inst.colorA = brightness;

                if (s.isHead) mw.headScratch[layer].push_back(inst);
                else          mw.trailScratch[layer].push_back(inst);
            }
        }
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(g_d3dContext->Map(g_viewportCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        float vp[4] = { static_cast<float>(mw.width), static_cast<float>(mw.height), 0.0f, 0.0f };
        memcpy(mapped.pData, vp, sizeof(vp));
        g_d3dContext->Unmap(g_viewportCB.Get(), 0);
    }

    UINT stride = sizeof(float) * 2, offset = 0;
    g_d3dContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    g_d3dContext->IASetInputLayout(g_inputLayout.Get());
    g_d3dContext->VSSetShader(g_vertexShader.Get(), nullptr, 0);
    g_d3dContext->VSSetConstantBuffers(0, 1, g_viewportCB.GetAddressOf());
    g_d3dContext->PSSetShader(g_pixelShader.Get(), nullptr, 0);
    g_d3dContext->PSSetSamplers(0, 1, g_samplerState.GetAddressOf());

    D3D11_VIEWPORT vp = { 0, 0, static_cast<float>(mw.width), static_cast<float>(mw.height), 0.0f, 1.0f };
    g_d3dContext->RSSetViewports(1, &vp);

    const float blendFactor[4] = { 0, 0, 0, 0 };
    g_d3dContext->OMSetRenderTargets(1, mw.trailRTV.GetAddressOf(), nullptr);
    g_d3dContext->OMSetBlendState(g_premulAlphaBlend.Get(), blendFactor, 0xFFFFFFFF);

    GlyphInstance fadeInst{};
    fadeInst.destX = 0; fadeInst.destY = 0;
    fadeInst.destW = static_cast<float>(mw.width);
    fadeInst.destH = static_cast<float>(mw.height);
    fadeInst.u0 = fadeInst.v0 = 0.0f; fadeInst.u1 = fadeInst.v1 = 1.0f;
    fadeInst.colorR = fadeInst.colorG = fadeInst.colorB = 0.0f;
    fadeInst.colorA = 0.16f;
    static thread_local std::vector<GlyphInstance> fadeScratch(1);
    fadeScratch[0] = fadeInst;
    UploadInstances(mw.fadeBuf, fadeScratch);
    DrawInstanced(g_whiteSRV.Get(), mw.fadeBuf, 1);

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const GlyphAtlas& atlas = g_atlases[layer];
        if (!mw.trailScratch[layer].empty()) {
            UploadInstances(mw.trailInstanceBuf[layer], mw.trailScratch[layer]);
            DrawInstanced(atlas.trailSRV.Get(), mw.trailInstanceBuf[layer], static_cast<UINT>(mw.trailScratch[layer].size()));
        }
        if (!mw.headScratch[layer].empty()) {
            UploadInstances(mw.headInstanceBuf[layer], mw.headScratch[layer]);
            DrawInstanced(atlas.headSRV.Get(), mw.headInstanceBuf[layer], static_cast<UINT>(mw.headScratch[layer].size()));
        }
    }

    ID3D11ShaderResourceView* nullSRV = nullptr;
    g_d3dContext->PSSetShaderResources(0, 1, &nullSRV);
    g_d3dContext->OMSetRenderTargets(1, mw.backBufferRTV.GetAddressOf(), nullptr);
    g_d3dContext->OMSetBlendState(g_opaqueBlend.Get(), blendFactor, 0xFFFFFFFF);

    GlyphInstance compositeInst{};
    compositeInst.destX = 0; compositeInst.destY = 0;
    compositeInst.destW = static_cast<float>(mw.width);
    compositeInst.destH = static_cast<float>(mw.height);
    compositeInst.u0 = compositeInst.v0 = 0.0f; compositeInst.u1 = compositeInst.v1 = 1.0f;
    compositeInst.colorR = compositeInst.colorG = compositeInst.colorB = compositeInst.colorA = 1.0f;
    static thread_local std::vector<GlyphInstance> compositeScratch(1);
    compositeScratch[0] = compositeInst;
    UploadInstances(mw.compositeBuf, compositeScratch);
    DrawInstanced(mw.trailSRV.Get(), mw.compositeBuf, 1);

    DrawBenchmarkOverlay(mw);

#if UNCAP_FRAMERATE
    // Vsync disabled: renders as fast as the GPU can produce frames, ignoring
    // the monitor's refresh rate. Useful for measuring true max throughput,
    // but will spin the GPU at high power/thermal cost for no visual benefit
    // (frames faster than the display can show are simply discarded/torn).
    HRESULT hr = mw.swapChain->Present(0, 0);
#else
    // Vsync enabled: Present blocks until the next vblank, which paces the
    // whole loop to this specific monitor's native refresh rate (e.g. 144Hz)
    // with no tearing and minimal wasted GPU work. Each monitor's swapchain
    // is presented independently, so mixed-refresh-rate setups (e.g. a 144Hz
    // primary next to a 60Hz secondary) each pace correctly against their own
    // display rather than being forced to a shared rate.
    HRESULT hr = mw.swapChain->Present(1, 0);
#endif
#if ENABLE_BENCHMARK_OVERLAY
    if (FAILED(hr)) ++g_droppedPresentCount;
#endif

    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        // Device loss affects every swapchain sharing the device, so the
        // full shared pipeline plus every window's resources must be rebuilt
        // together -- not just this one monitor's.
        DiscardAllDeviceResources();
        const wchar_t* failedStage = nullptr;
        if (SUCCEEDED(InitSharedPipeline(&failedStage))) {
            for (auto& other : g_monitorWindows) {
                if (other && SUCCEEDED(InitMonitorWindow(*other))) {
                    InitColumns(*other, other->width, other->height);
                }
            }
            if (g_previewWindow && SUCCEEDED(InitMonitorWindow(*g_previewWindow))) {
                InitColumns(*g_previewWindow, g_previewWindow->width, g_previewWindow->height);
            }
        }
    }
}

static void ResetMouseTracking(HWND hwnd) {
    GetCursorPos(&g_lastMousePos);
    g_mouseInit = true;
}

// ---------------------------------------------------------------------------
// Multi-monitor enumeration
// ---------------------------------------------------------------------------
// EnumDisplayMonitors callback: records each active display's virtual-desktop
// geometry (which correctly handles negative coordinates for monitors placed
// left-of/above the primary, and arbitrary width/height for mixed portrait/
// landscape or mixed-resolution setups) plus its HMONITOR handle and
// primary-monitor flag.
static BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC /*hdcMonitor*/, LPRECT /*lprcMonitor*/, LPARAM lParam) {
    auto* outList = reinterpret_cast<std::vector<std::unique_ptr<MonitorWindow>>*>(lParam);

    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hMonitor, &mi)) return TRUE; // skip this monitor, keep enumerating

    auto mw = std::make_unique<MonitorWindow>();
    mw->hMonitor = hMonitor;
    // rcMonitor (not rcWork) is the *entire* physical display surface,
    // including any taskbar area -- exactly what a borderless fullscreen
    // screensaver window should cover. Coordinates are in virtual-desktop
    // space, so a monitor to the left of the primary will have negative x.
    mw->x = mi.rcMonitor.left;
    mw->y = mi.rcMonitor.top;
    mw->width = mi.rcMonitor.right - mi.rcMonitor.left;
    mw->height = mi.rcMonitor.bottom - mi.rcMonitor.top;
    mw->isPrimary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;

    outList->push_back(std::move(mw));
    return TRUE; // continue enumerating remaining monitors
}

// Populates g_monitorWindows with one MonitorWindow per active display,
// using EnumDisplayMonitors to capture exact per-monitor geometry. Safe to
// call multiple times (e.g. on WM_DISPLAYCHANGE) -- existing entries are
// cleared first. Does not create any HWND/D3D resources; that happens
// afterward in CreateAllMonitorWindows().
static void EnumerateActiveMonitors(std::vector<std::unique_ptr<MonitorWindow>>& outList) {
    outList.clear();
    EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, reinterpret_cast<LPARAM>(&outList));

    // Put the primary monitor first for a stable, predictable render order
    // (mainly cosmetic -- e.g. so the primary shows the benchmark overlay
    // first in any log output -- but also gives deterministic Z/topmost
    // ordering among the windows we're about to create).
    std::stable_sort(outList.begin(), outList.end(),
        [](const std::unique_ptr<MonitorWindow>& a, const std::unique_ptr<MonitorWindow>& b) {
            return a->isPrimary && !b->isPrimary;
        });
}

// ---------------------------------------------------------------------------
// Messaging System Window Pipelines
// ---------------------------------------------------------------------------
// A single WndProc serves every monitor's fullscreen HWND plus the preview
// HWND. Each window's associated MonitorWindow* is stashed in GWLP_USERDATA
// at WM_NCCREATE/WM_CREATE time (via CREATESTRUCT::lpCreateParams, which we
// populate ourselves in CreateWindowExW's lpParam argument), so every
// message can be routed to the right per-window state without any global
// "current window" assumption.
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    // Stash the MonitorWindow* pointer (passed as CreateWindowExW's lpParam)
    // into GWLP_USERDATA as soon as the window is created, so every
    // subsequent message can retrieve it with GetWindowLongPtr.
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        // Set MonitorWindow::hwnd here, immediately, rather than waiting for
        // CreateWindowExW to return. WS_VISIBLE windows dispatch WM_CREATE
        // synchronously from inside CreateWindowExW, so anything done in
        // WM_CREATE (including swap-chain creation, which needs a valid
        // OutputWindow) would otherwise see a null mw->hwnd and fail with
        // DXGI_ERROR_INVALID_CALL.
        auto* earlyMw = reinterpret_cast<MonitorWindow*>(cs->lpCreateParams);
        if (earlyMw) earlyMw->hwnd = hwnd;
        // Fall through to DefWindowProc for the actual WM_NCCREATE handling.
    }

    MonitorWindow* mw = reinterpret_cast<MonitorWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_CREATE: {
        if (!mw) return -1; // should never happen; abort window creation

        RECT rc;
        GetClientRect(hwnd, &rc);
        mw->width = rc.right - rc.left;
        mw->height = rc.bottom - rc.top;

        // The shared device/shader/atlas pipeline (plus process-global
        // facilities like the PDH performance counters, which must only be
        // opened once regardless of how many monitor windows exist) is
        // initialized exactly once, the first time any window is created
        // (fullscreen monitor window or preview). Subsequent windows only
        // need their own swapchain and per-window resources against the
        // already-live device.
        static bool s_sharedPipelineReady = false;
        if (!s_sharedPipelineReady) {
            SeedFastRng(static_cast<uint32_t>(time(nullptr)) ^ static_cast<uint32_t>(GetTickCount64()));
            g_startTick = GetTickCount64();

            const wchar_t* failedStage = L"(unknown)";
            HRESULT initHr = InitSharedPipeline(&failedStage);
            if (FAILED(initHr)) {
                wchar_t buf[192];
                swprintf_s(buf, L"Direct3D11 shared pipeline initialization failed.\nStage: %s\nHRESULT: 0x%08X",
                    failedStage, static_cast<unsigned int>(initHr));
                MessageBoxW(hwnd, buf, L"Matrix Screensaver", MB_OK | MB_ICONERROR);
                return -1;
            }
            InitPerfCounters();
            s_sharedPipelineReady = true;
        }

        const wchar_t* failedStage = L"(unknown)";
        HRESULT initHr = InitMonitorWindow(*mw, &failedStage);
        if (FAILED(initHr)) {
            wchar_t buf[192];
            swprintf_s(buf, L"Direct3D11 window initialization failed.\nStage: %s\nHRESULT: 0x%08X", failedStage, static_cast<unsigned int>(initHr));
            MessageBoxW(hwnd, buf, L"Matrix Screensaver", MB_OK | MB_ICONERROR);
            return -1;
        }

        InitColumns(*mw, mw->width, mw->height);
        ++g_liveWindowCount;

        if (!g_isPreview) ShowCursor(FALSE);
        ResetMouseTracking(hwnd);

        // The small embedded preview window (screensaver picker thumbnail) has
        // no reason to render at full monitor refresh rate -- it's tiny and
        // usually not even visible for long. Keep it on a modest timer so it
        // doesn't compete for GPU/CPU with whatever else is running. Fullscreen
        // monitor windows are instead driven by the main PeekMessage loop in
        // wWinMain, paced by each window's own Present()'s vsync wait, so no
        // timer is needed there.
        if (g_isPreview) {
            SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL_MS, nullptr);
        }
        return 0;
    }
    case WM_TIMER: {
        if (mw && mw->isVisible) DrawFrame(*mw, static_cast<DWORD>(GetTickCount64()));
        return 0;
    }
    case WM_SHOWWINDOW: {
        if (mw) mw->isVisible = (wParam != 0);
        return 0;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_SIZE: {
        if (!mw) break;
        mw->isVisible = (wParam != SIZE_MINIMIZED);
        if (mw->swapChain) {
            RECT rc;
            GetClientRect(hwnd, &rc);
            int newWidth = rc.right - rc.left;
            int newHeight = rc.bottom - rc.top;
            if (newWidth < 1) newWidth = 1;
            if (newHeight < 1) newHeight = 1;

            if (newWidth != mw->width || newHeight != mw->height) {
                mw->width = newWidth;
                mw->height = newHeight;

                g_d3dContext->OMSetRenderTargets(0, nullptr, nullptr);
                mw->backBufferRTV.Reset();
                mw->trailRTV.Reset();
                mw->trailSRV.Reset();
                mw->trailTexture.Reset();
#if ENABLE_BENCHMARK_OVERLAY
                mw->overlayD2DTarget.Reset();
#endif

                HRESULT hr = mw->swapChain->ResizeBuffers(0, static_cast<UINT>(mw->width),
                    static_cast<UINT>(mw->height), DXGI_FORMAT_UNKNOWN, 0);
                if (SUCCEEDED(hr)) {
                    hr = CreateSizeDependentResources(*mw, mw->width, mw->height);
                }

                if (FAILED(hr)) {
                    // Resize failed (e.g. device lost mid-resize): rebuild
                    // this window's resources against the existing shared
                    // device. If the device itself is gone, DrawFrame's own
                    // DXGI_ERROR_DEVICE_REMOVED/RESET handling will catch it
                    // on the next frame and rebuild everything.
                    DiscardWindowResources(*mw);
                    if (SUCCEEDED(InitMonitorWindow(*mw))) {
                        InitColumns(*mw, mw->width, mw->height);
                    }
                    return 0;
                }

                InitColumns(*mw, mw->width, mw->height);
            }
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
            // Any monitor's window exiting on mouse movement should end the
            // whole multi-monitor screensaver session, not just that one
            // window -- otherwise the user could be left with N-1 fullscreen
            // black windows still covering their other monitors. Destroying
            // every window here causes each to individually hit WM_DESTROY
            // and PostQuitMessage, which is harmless (PostQuitMessage can be
            // called multiple times; the loop exits on the first WM_QUIT it
            // sees).
            for (auto& other : g_monitorWindows) {
                if (other && other->hwnd && IsWindow(other->hwnd)) DestroyWindow(other->hwnd);
            }
        }
        return 0;
    }
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
        if (!g_isPreview && !InGracePeriod()) {
            for (auto& other : g_monitorWindows) {
                if (other && other->hwnd && IsWindow(other->hwnd)) DestroyWindow(other->hwnd);
            }
        }
        return 0;

    case WM_KEYDOWN:
        // 'B' toggles the benchmark/diagnostics overlay without exiting the
        // screensaver, so you can check performance without losing the session.
        // Only meaningful when the overlay is compiled in; when
        // ENABLE_BENCHMARK_OVERLAY is 0, 'B' falls through and exits like any
        // other key, since there's nothing to toggle.
#if ENABLE_BENCHMARK_OVERLAY
        if (wParam == 'B') {
            g_benchEnabled = !g_benchEnabled;
            return 0;
        }
#endif
        if (!g_isPreview && !InGracePeriod()) {
            for (auto& other : g_monitorWindows) {
                if (other && other->hwnd && IsWindow(other->hwnd)) DestroyWindow(other->hwnd);
            }
        }
        return 0;

    case WM_ACTIVATE:
        // Only the specific window that lost activation should trigger
        // exit-on-deactivate; with several topmost fullscreen windows,
        // clicking from one monitor to another would otherwise immediately
        // tear down the whole session as soon as focus moved between them.
        // Real user-initiated deactivation (Alt+Tab away, another app
        // stealing focus) still exits normally via this same path.
        if (!g_isPreview && !InGracePeriod() && LOWORD(wParam) == WA_INACTIVE) {
            HWND newFocus = reinterpret_cast<HWND>(lParam);
            bool activatingAnotherMonitorWindow = false;
            for (auto& other : g_monitorWindows) {
                if (other && other->hwnd == newFocus) { activatingAnotherMonitorWindow = true; break; }
            }
            if (!activatingAnotherMonitorWindow) {
                for (auto& other : g_monitorWindows) {
                    if (other && other->hwnd && IsWindow(other->hwnd)) DestroyWindow(other->hwnd);
                }
            }
        }
        return 0;

    case WM_DESTROY:
        KillTimer(hwnd, TIMER_ID);
        if (!g_isPreview) ShowCursor(TRUE);
        if (mw) {
            DiscardWindowResources(*mw);
        }
        // Only tear down the shared device/pipeline and process-global
        // facilities (PDH perf counters) once every monitor window (and the
        // preview window, if any) has been destroyed -- per-window
        // WM_DESTROY must not kill resources other windows still depend on.
        // g_liveWindowCount is incremented once per successful WM_CREATE, so
        // it reaches zero exactly when the last live window goes away.
        // wWinMain also calls DiscardSharedDeviceResources() itself after the
        // message loop exits as a safety net; that call is idempotent.
        if (--g_liveWindowCount <= 0) {
            ShutdownPerfCounters();
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Creates one borderless, topmost, fullscreen HWND per entry already
// populated in g_monitorWindows (see EnumerateActiveMonitors), positioned
// and sized to exactly match that monitor's virtual-desktop rectangle. This
// correctly handles asymmetrical setups -- differing resolutions, portrait/
// landscape mixes, and monitors offset to the left of/above the primary
// (negative coordinates) -- because it uses each MonitorWindow's own x/y/
// width/height rather than any single GetSystemMetrics(SM_CXSCREEN)-style
// primary-only value.
//
// Returns true only if every monitor window was created successfully; on
// partial failure, any windows already created are left intact (destroying
// them here would be premature since the caller may choose to continue with
// a reduced set), but the caller should treat an overall false as fatal.
static bool CreateAllMonitorWindows(HINSTANCE hInstance) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"MatrixScreensaverClass";
    wc.hCursor = nullptr;
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassW(&wc);

    for (auto& mwPtr : g_monitorWindows) {
        MonitorWindow* mw = mwPtr.get();

        // lpCreateParams (final argument) is retrieved inside WndProc's
        // WM_NCCREATE handler and stashed into GWLP_USERDATA, which is how
        // this single shared WndProc tells which MonitorWindow a given HWND
        // belongs to for every later message.
        HWND hwnd = CreateWindowExW(
            WS_EX_TOPMOST, wc.lpszClassName, L"Matrix Screensaver",
            WS_POPUP | WS_VISIBLE,
            mw->x, mw->y, mw->width, mw->height,
            nullptr, nullptr, hInstance, mw);

        if (!hwnd) return false;
        mw->hwnd = hwnd;
    }
    return true;
}

static HWND CreatePreviewWindow(HINSTANCE hInstance, HWND parent, MonitorWindow* mw) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"MatrixScreensaverPreviewClass";
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassW(&wc);

    RECT rc;
    GetClientRect(parent, &rc);
    return CreateWindowExW(0, wc.lpszClassName, L"", WS_CHILD | WS_VISIBLE,
        0, 0, rc.right - rc.left, rc.bottom - rc.top, parent, nullptr, hInstance, mw);
}

static void ShowConfigDialog(HINSTANCE hInstance, HWND ownerHwnd) {
    MessageBoxW(ownerHwnd,
        L"Matrix Screensaver\n\nNo configurable options.",
        L"Settings", MB_OK | MB_ICONINFORMATION);
}

// ---------------------------------------------------------------------------
// Runtime Entry Directives
// ---------------------------------------------------------------------------
int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(nCmdShow);

    // Declare Per-Monitor-V2 DPI awareness *before* creating any window or
    // calling GetClientRect. Without this, an unaware process gets silently
    // DPI-virtualized by Windows: the OS renders our window into a smaller
    // backing buffer sized for 96 DPI and then bitmap-stretches it to fit
    // the "logical" (scaled) window rect it told us about. That mismatch is
    // especially visible for the screensaver picker's preview thumbnail --
    // a child window embedded inside a foreign, DPI-aware Control Panel
    // dialog -- where our unaware child ends up rendering into a fraction of
    // the space the dialog actually gave it (the rest is simply left
    // unpainted, appearing as a dark region). Declaring awareness up front
    // makes GetClientRect/CreateWindowExW/D3D swap chain sizes all agree in
    // real physical pixels, with no OS-level stretching anywhere.
    //
    // SetProcessDpiAwarenessContext is the modern API (Windows 10 1703+).
    // Fall back to SetProcessDpiAwareness (8.1+) and finally
    // SetProcessDPIAware (Vista+) for robustness on older systems, since this
    // ships as a plain .scr/.exe without an application manifest declaring
    // DPI awareness statically.
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    using SetCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    auto setCtx = user32 ? reinterpret_cast<SetCtxFn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext")) : nullptr;
    if (setCtx && setCtx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        // Success on Windows 10 1703+.
    }
    else {
        HMODULE shcore = LoadLibraryW(L"Shcore.dll");
        using SetAwarenessFn = HRESULT(WINAPI*)(PROCESS_DPI_AWARENESS);
        auto setAwareness = shcore ? reinterpret_cast<SetAwarenessFn>(GetProcAddress(shcore, "SetProcessDpiAwareness")) : nullptr;
        if (setAwareness) {
            setAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
        }
        else {
            SetProcessDPIAware();
        }
        if (shcore) FreeLibrary(shcore);
    }

    std::wstring cmd(lpCmdLine ? lpCmdLine : L"");
    auto lower = cmd;
    for (auto& ch : lower) ch = towlower(ch);

    bool isConfig = false, isPreview = false;
    HWND targetParent = nullptr;

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
        g_previewWindow = std::make_unique<MonitorWindow>();
        HWND previewHwnd = CreatePreviewWindow(hInstance, targetParent, g_previewWindow.get());
        if (!previewHwnd) return 0;
        g_previewWindow->hwnd = previewHwnd;

        ShowWindow(previewHwnd, SW_SHOW);
        UpdateWindow(previewHwnd);

        // Preview mode is driven entirely by its own WM_TIMER (set up in
        // WM_CREATE), so this just needs a standard blocking message pump.
        MSG msg{};
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        DiscardSharedDeviceResources();
        return static_cast<int>(msg.wParam);
    }

    // Full multi-monitor fullscreen mode: enumerate every active display via
    // EnumDisplayMonitors, then create one independent borderless fullscreen
    // window per monitor, each sized/positioned to that monitor's exact
    // virtual-desktop rectangle (correctly handling negative offsets,
    // mismatched resolutions, and portrait/landscape mixes).
    g_isPreview = false;
    EnumerateActiveMonitors(g_monitorWindows);

    if (g_monitorWindows.empty()) {
        // Should be unreachable on any real system (there's always at least
        // one display), but fall back to treating it as a single-monitor
        // failure rather than silently doing nothing.
        MessageBoxW(nullptr, L"No active displays were detected.", L"Matrix Screensaver", MB_OK | MB_ICONERROR);
        return 0;
    }

    if (!CreateAllMonitorWindows(hInstance)) {
        MessageBoxW(nullptr, L"Failed to create one or more monitor windows.", L"Matrix Screensaver", MB_OK | MB_ICONERROR);
        // Tear down anything already created via each HWND's own WM_DESTROY.
        for (auto& mwPtr : g_monitorWindows) {
            if (mwPtr && mwPtr->hwnd) DestroyWindow(mwPtr->hwnd);
        }
        return 0;
    }

    for (auto& mwPtr : g_monitorWindows) {
        ShowWindow(mwPtr->hwnd, SW_SHOW);
        UpdateWindow(mwPtr->hwnd);
    }

    // Multi-monitor render loop: drain all pending Windows messages (across
    // every monitor window) without blocking, then render+present one frame
    // on each visible monitor window in turn. Each window's own
    // IDXGISwapChain::Present(1,0) call (vsync on) blocks until that specific
    // monitor's next vblank, so a mixed-refresh-rate setup (e.g. 144Hz
    // primary + 60Hz secondary) naturally paces each swapchain against its
    // own display rather than forcing a single shared rate. The loop as a
    // whole is gated by the slowest visible monitor in any given pass, which
    // is the same trade-off any multi-head fullscreen D3D11 app makes.
    MSG msg{};
    bool running = true;
    while (running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!running) break;

        bool anyVisible = false;
        for (auto& mwPtr : g_monitorWindows) {
            if (mwPtr && mwPtr->isVisible) {
                DrawFrame(*mwPtr, static_cast<DWORD>(GetTickCount64()));
                anyVisible = true;
            }
        }
        if (!anyVisible) {
            // Nothing to render right now (all monitor windows hidden or
            // minimized) -- avoid a hot spin loop burning a CPU core for no
            // reason.
            WaitMessage();
        }
    }

    // The message loop only exits once every monitor window has posted
    // WM_QUIT via its own WM_DESTROY (each window's WM_DESTROY handler tears
    // down just that window's own resources). Now that all windows are gone,
    // it's safe to release the shared device/pipeline/atlases.
    DiscardSharedDeviceResources();
    g_monitorWindows.clear();

    return static_cast<int>(msg.wParam);
}