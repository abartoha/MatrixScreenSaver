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
#include <string>
#include <deque>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <windows.h>
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

static HWND  g_hwnd = nullptr;
static int   g_width = 0, g_height = 0;
static std::vector<ColumnState> g_columns;
static std::vector<std::vector<size_t>> g_columnsByLayer;

// ---------------------------------------------------------------------------
// Direct3D11 Pipeline Core
// ---------------------------------------------------------------------------
static ComPtr<ID3D11Device>           g_d3dDevice;
static ComPtr<ID3D11DeviceContext>    g_d3dContext;
static ComPtr<IDXGISwapChain>         g_swapChain;
static ComPtr<ID3D11RenderTargetView> g_backBufferRTV;

static ComPtr<ID3D11Texture2D>          g_trailTexture;
static ComPtr<ID3D11RenderTargetView>   g_trailRTV;
static ComPtr<ID3D11ShaderResourceView> g_trailSRV;

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
static GlyphAtlas g_atlases[NUM_LAYERS];

struct LayerInstanceBuffer {
    ComPtr<ID3D11Buffer> buffer;
    UINT capacity = 0;
};
static LayerInstanceBuffer g_headInstanceBuf[NUM_LAYERS];
static LayerInstanceBuffer g_trailInstanceBuf[NUM_LAYERS];

static std::vector<GlyphInstance> g_headScratch[NUM_LAYERS];
static std::vector<GlyphInstance> g_trailScratch[NUM_LAYERS];

static bool g_isPreview = false;
static ULONGLONG g_startTick = 0;
static const ULONGLONG STARTUP_GRACE_MS = 1000;
static bool InGracePeriod() { return (GetTickCount64() - g_startTick) < STARTUP_GRACE_MS; }

static POINT g_lastMousePos{};
static bool  g_mouseInit = false;
static bool  g_isVisible = true;

// ---------------------------------------------------------------------------
// Benchmark / Diagnostics Overlay
// ---------------------------------------------------------------------------
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

// D2D text resources for the overlay (created alongside other size-dependent resources)
static ComPtr<IDWriteTextFormat>    g_overlayTextFormat;
static ComPtr<ID2D1SolidColorBrush> g_overlayTextBrush;
static ComPtr<ID2D1SolidColorBrush> g_overlayBgBrush;
static ComPtr<ID2D1Bitmap1>         g_overlayD2DTarget; // D2D view of the backbuffer, for direct overlay draw

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

static void InitColumns(int width, int height) {
    g_columns.clear();
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
            int startY = (height > 0) ? -static_cast<int>(FastRandBounded(static_cast<uint32_t>(height))) : 0;

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
            g_columns.push_back(std::move(col));
        }
    }

    g_columnsByLayer.assign(NUM_LAYERS, {});
    for (size_t i = 0; i < g_columns.size(); ++i) {
        g_columnsByLayer[g_columns[i].layer].push_back(i);
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

// ---------------------------------------------------------------------------
// Direct3D11 Setup Execution Contracts
// ---------------------------------------------------------------------------
static HRESULT CreateD3DDeviceAndSwapChain(HWND hwnd, int width, int height) {
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
        hr = D3D11CreateDeviceAndSwapChain(
            explicitAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags | debugFlag,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
        if (FAILED(hr) && debugFlag != 0) {
            hr = D3D11CreateDeviceAndSwapChain(
                explicitAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags,
                levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
        }
    }
    else {
        // Typo fix applied: singular g_d3dContext
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags | debugFlag,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);

        if (FAILED(hr) && debugFlag != 0) {
            hr = D3D11CreateDeviceAndSwapChain(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags,
                levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
        }
    }

    if (FAILED(hr)) {
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, deviceFlags,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
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
        if (factory1) EnumerateAndSelectAdapter(factory1.Get(), actualAdapter.Get());
        InitGpuMemoryQuery();
    }

    return hr;
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

static HRESULT CreateSizeDependentResources(int width, int height) {
    ComPtr<ID3D11Texture2D> backBuffer;
    HRESULT hr = g_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateRenderTargetView(backBuffer.Get(), nullptr, &g_backBufferRTV);
    if (FAILED(hr)) return hr;

    D3D11_TEXTURE2D_DESC ttd = {};
    ttd.Width = width; ttd.Height = height; ttd.MipLevels = 1; ttd.ArraySize = 1;
    ttd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    ttd.SampleDesc.Count = 1;
    ttd.Usage = D3D11_USAGE_DEFAULT;
    ttd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    hr = g_d3dDevice->CreateTexture2D(&ttd, nullptr, &g_trailTexture);
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateRenderTargetView(g_trailTexture.Get(), nullptr, &g_trailRTV);
    if (FAILED(hr)) return hr;
    hr = g_d3dDevice->CreateShaderResourceView(g_trailTexture.Get(), nullptr, &g_trailSRV);
    if (FAILED(hr)) return hr;

    const float black[4] = { 0, 0, 0, 1 };
    g_d3dContext->ClearRenderTargetView(g_trailRTV.Get(), black);

    // D2D bitmap view onto the swap chain's backbuffer surface, used only to
    // draw the benchmark overlay text on top of the composited frame.
    g_overlayD2DTarget.Reset();
    if (g_d2dContext) {
        ComPtr<IDXGISurface> backSurface;
        if (SUCCEEDED(backBuffer.As(&backSurface))) {
            D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
                D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
            g_d2dContext->CreateBitmapFromDxgiSurface(backSurface.Get(), &bp, &g_overlayD2DTarget);
        }
    }
    return S_OK;
}

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

static void DiscardDeviceResources() {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        g_atlases[layer].headTexture.Reset();
        g_atlases[layer].trailTexture.Reset();
        g_atlases[layer].headSRV.Reset();
        g_atlases[layer].trailSRV.Reset();
        g_headInstanceBuf[layer].buffer.Reset();
        g_headInstanceBuf[layer].capacity = 0;
        g_trailInstanceBuf[layer].buffer.Reset();
        g_trailInstanceBuf[layer].capacity = 0;
    }
    g_trailTexture.Reset();
    g_trailRTV.Reset();
    g_trailSRV.Reset();
    g_backBufferRTV.Reset();
    g_overlayD2DTarget.Reset();
    g_overlayTextFormat.Reset();
    g_overlayTextBrush.Reset();
    g_overlayBgBrush.Reset();
    g_dxgiAdapter3.Reset();
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
    g_swapChain.Reset();
    g_d3dContext.Reset();
    g_d3dDevice.Reset();
}

static HRESULT InitDirect2D(HWND hwnd, const wchar_t** failedStage = nullptr) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    int width = rc.right - rc.left, height = rc.bottom - rc.top;
    if (width < 1) width = 1;
    if (height < 1) height = 1;

    HRESULT hr = CreateD3DDeviceAndSwapChain(hwnd, width, height);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateD3DDeviceAndSwapChain"; return hr; }

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

    hr = CreateSizeDependentResources(width, height);
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateSizeDependentResources"; return hr; }

    hr = CreateOverlayResources();
    if (FAILED(hr)) { if (failedStage) *failedStage = L"CreateOverlayResources"; return hr; }

    QueryPerformanceFrequency(&g_qpcFrequency);
    QueryPerformanceCounter(&g_lastFrameQpc);
    g_frameTimesMs.clear();
    g_totalFramesRendered = 0;
    g_droppedPresentCount = 0;

    for (int i = 0; i < NUM_LAYERS; ++i) {
        hr = BuildAtlasForLayer(i);
        if (FAILED(hr)) { if (failedStage) *failedStage = L"BuildAtlasForLayer"; return hr; }
    }
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
static void DrawBenchmarkOverlay() {
    if (!g_benchEnabled || !g_d2dContext || !g_overlayD2DTarget || !g_overlayTextFormat) return;

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

    g_d2dContext->SetTarget(g_overlayD2DTarget.Get());
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

static void DrawFrame(DWORD tickCount) {
    if (!g_d3dContext || !g_trailRTV || !g_backBufferRTV) return;

    LARGE_INTEGER frameStartQpc;
    QueryPerformanceCounter(&frameStartQpc);
    if (g_qpcFrequency.QuadPart > 0) {
        double deltaMs = static_cast<double>(frameStartQpc.QuadPart - g_lastFrameQpc.QuadPart) * 1000.0
            / static_cast<double>(g_qpcFrequency.QuadPart);
        g_frameTimesMs.push_back(deltaMs);
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

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        g_headScratch[layer].clear();
        g_trailScratch[layer].clear();
    }

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerConfig& cfg = LAYER_CONFIGS[layer];
        const GlyphAtlas& atlas = g_atlases[layer];
        const float cellF = static_cast<float>(atlas.cellWidth);
        const float brightness = cfg.brightnessMul;

        for (size_t colIdx : g_columnsByLayer[layer]) {
            ColumnState& col = g_columns[colIdx];

            for (auto& s : col.symbols) {
                if (s.interval > 0 && tickCount >= s.nextChangeTick) {
                    s.value = RandomKatakana();
                    s.nextChangeTick = tickCount + static_cast<DWORD>(s.interval);
                }

                s.y += static_cast<float>(s.speed) * cfg.speedMul * 0.6f;
                if (s.y > g_height) s.y = -static_cast<float>(cfg.fontSize);

                if (s.y + cellF < 0.0f || s.y > static_cast<float>(g_height)) continue;

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

                if (s.isHead) g_headScratch[layer].push_back(inst);
                else          g_trailScratch[layer].push_back(inst);
            }
        }
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(g_d3dContext->Map(g_viewportCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        float vp[4] = { static_cast<float>(g_width), static_cast<float>(g_height), 0.0f, 0.0f };
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

    D3D11_VIEWPORT vp = { 0, 0, static_cast<float>(g_width), static_cast<float>(g_height), 0.0f, 1.0f };
    g_d3dContext->RSSetViewports(1, &vp);

    const float blendFactor[4] = { 0, 0, 0, 0 };
    g_d3dContext->OMSetRenderTargets(1, g_trailRTV.GetAddressOf(), nullptr);
    g_d3dContext->OMSetBlendState(g_premulAlphaBlend.Get(), blendFactor, 0xFFFFFFFF);

    GlyphInstance fadeInst{};
    fadeInst.destX = 0; fadeInst.destY = 0;
    fadeInst.destW = static_cast<float>(g_width);
    fadeInst.destH = static_cast<float>(g_height);
    fadeInst.u0 = fadeInst.v0 = 0.0f; fadeInst.u1 = fadeInst.v1 = 1.0f;
    fadeInst.colorR = fadeInst.colorG = fadeInst.colorB = 0.0f;
    fadeInst.colorA = 0.16f;
    static std::vector<GlyphInstance> fadeScratch(1);
    fadeScratch[0] = fadeInst;
    static LayerInstanceBuffer fadeBuf;
    UploadInstances(fadeBuf, fadeScratch);
    DrawInstanced(g_whiteSRV.Get(), fadeBuf, 1);

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const GlyphAtlas& atlas = g_atlases[layer];
        if (!g_trailScratch[layer].empty()) {
            UploadInstances(g_trailInstanceBuf[layer], g_trailScratch[layer]);
            DrawInstanced(atlas.trailSRV.Get(), g_trailInstanceBuf[layer], static_cast<UINT>(g_trailScratch[layer].size()));
        }
        if (!g_headScratch[layer].empty()) {
            UploadInstances(g_headInstanceBuf[layer], g_headScratch[layer]);
            DrawInstanced(atlas.headSRV.Get(), g_headInstanceBuf[layer], static_cast<UINT>(g_headScratch[layer].size()));
        }
    }

    ID3D11ShaderResourceView* nullSRV = nullptr;
    g_d3dContext->PSSetShaderResources(0, 1, &nullSRV);
    g_d3dContext->OMSetRenderTargets(1, g_backBufferRTV.GetAddressOf(), nullptr);
    g_d3dContext->OMSetBlendState(g_opaqueBlend.Get(), blendFactor, 0xFFFFFFFF);

    GlyphInstance compositeInst{};
    compositeInst.destX = 0; compositeInst.destY = 0;
    compositeInst.destW = static_cast<float>(g_width);
    compositeInst.destH = static_cast<float>(g_height);
    compositeInst.u0 = compositeInst.v0 = 0.0f; compositeInst.u1 = compositeInst.v1 = 1.0f;
    compositeInst.colorR = compositeInst.colorG = compositeInst.colorB = compositeInst.colorA = 1.0f;
    static std::vector<GlyphInstance> compositeScratch(1);
    compositeScratch[0] = compositeInst;
    static LayerInstanceBuffer compositeBuf;
    UploadInstances(compositeBuf, compositeScratch);
    DrawInstanced(g_trailSRV.Get(), compositeBuf, 1);

    DrawBenchmarkOverlay();

    HRESULT hr = g_swapChain->Present(1, 0);
    if (FAILED(hr)) ++g_droppedPresentCount;

    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        DiscardDeviceResources();
        if (SUCCEEDED(InitDirect2D(g_hwnd))) {
            InitColumns(g_width, g_height);
        }
    }
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

        const wchar_t* failedStage = L"(unknown)";
        HRESULT initHr = InitDirect2D(hwnd, &failedStage);
        if (FAILED(initHr)) {
            wchar_t msg[192];
            swprintf_s(msg, L"Direct2D/D3D11 initialization failed.\nStage: %s\nHRESULT: 0x%08X", failedStage, static_cast<unsigned int>(initHr));
            MessageBoxW(hwnd, msg, L"Matrix Screensaver", MB_OK | MB_ICONERROR);
            DestroyWindow(hwnd);
            return 0;
        }

        InitColumns(g_width, g_height);
        InitPerfCounters();

        if (!g_isPreview) ShowCursor(FALSE);
        ResetMouseTracking(hwnd);
        SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL_MS, nullptr);
        return 0;
    }
    case WM_TIMER: {
        if (g_isVisible) DrawFrame(static_cast<DWORD>(GetTickCount64()));
        return 0;
    }
    case WM_SHOWWINDOW: {
        g_isVisible = (wParam != 0);
        return 0;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_SIZE: {
        g_isVisible = (wParam != SIZE_MINIMIZED);
        if (g_swapChain) {
            RECT rc;
            GetClientRect(hwnd, &rc);
            int newWidth = rc.right - rc.left;
            int newHeight = rc.bottom - rc.top;
            if (newWidth < 1) newWidth = 1;
            if (newHeight < 1) newHeight = 1;

            if (newWidth != g_width || newHeight != g_height) {
                g_width = newWidth;
                g_height = newHeight;

                g_d3dContext->OMSetRenderTargets(0, nullptr, nullptr);
                g_backBufferRTV.Reset();
                g_trailRTV.Reset();
                g_trailSRV.Reset();
                g_trailTexture.Reset();

                HRESULT hr = g_swapChain->ResizeBuffers(0, static_cast<UINT>(g_width),
                    static_cast<UINT>(g_height), DXGI_FORMAT_UNKNOWN, 0);
                if (SUCCEEDED(hr)) {
                    hr = CreateSizeDependentResources(g_width, g_height);
                }

                if (FAILED(hr)) {
                    DiscardDeviceResources();
                    if (SUCCEEDED(InitDirect2D(hwnd))) {
                        InitColumns(g_width, g_height);
                    }
                    return 0;
                }

                InitColumns(g_width, g_height);
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
            DestroyWindow(hwnd);
        }
        return 0;
    }
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
        if (!g_isPreview && !InGracePeriod()) DestroyWindow(hwnd);
        return 0;

    case WM_KEYDOWN:
        // 'B' toggles the benchmark/diagnostics overlay without exiting the
        // screensaver, so you can check performance without losing the session.
        // Only honored in preview/windowed contexts or during the grace period
        // isn't required here since toggling isn't destructive; any other key
        // still exits as before.
        if (wParam == 'B') {
            g_benchEnabled = !g_benchEnabled;
            return 0;
        }
        if (!g_isPreview && !InGracePeriod()) DestroyWindow(hwnd);
        return 0;

    case WM_ACTIVATE:
        if (!g_isPreview && !InGracePeriod() && LOWORD(wParam) == WA_INACTIVE) DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        KillTimer(hwnd, TIMER_ID);
        if (!g_isPreview) ShowCursor(TRUE);
        ShutdownPerfCounters();
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
        L"Matrix Screensaver\n\nNo configurable options.",
        L"Settings", MB_OK | MB_ICONINFORMATION);
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