// MatrixScreensaver.cpp
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
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <windows.h>
#include <d2d1.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
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
// Column indices bucketed per layer so DrawFrame can iterate each column
// exactly once per frame instead of scanning the full column list once
// per layer (was O(NUM_LAYERS * columns), now O(columns)).
static std::vector<std::vector<size_t>> g_columnsByLayer;

// ---------------------------------------------------------------------------
// Direct3D11 Pipeline Core
// ---------------------------------------------------------------------------
// One instanced draw call renders every visible glyph for a given
// (layer, head-or-trail) atlas in a single GPU submission, replacing what
// used to be one DrawBitmap call per visible glyph.
static ComPtr<ID3D11Device>           g_d3dDevice;
static ComPtr<ID3D11DeviceContext>    g_d3dContext;
static ComPtr<IDXGISwapChain>         g_swapChain;
static ComPtr<ID3D11RenderTargetView> g_backBufferRTV;

// Persistent offscreen trail buffer (equivalent to the old g_trailTarget):
// never cleared between frames, only faded via a translucent black quad,
// so the falling-glyph trails accumulate exactly as before.
static ComPtr<ID3D11Texture2D>          g_trailTexture;
static ComPtr<ID3D11RenderTargetView>   g_trailRTV;
static ComPtr<ID3D11ShaderResourceView> g_trailSRV;

static ComPtr<ID3D11VertexShader>  g_vertexShader;
static ComPtr<ID3D11PixelShader>   g_pixelShader;
static ComPtr<ID3D11InputLayout>   g_inputLayout;
static ComPtr<ID3D11Buffer>        g_quadVertexBuffer;   // static unit quad, 4 verts
static ComPtr<ID3D11Buffer>        g_viewportCB;          // b0: viewport size
static ComPtr<ID3D11SamplerState>  g_samplerState;
static ComPtr<ID3D11BlendState>    g_premulAlphaBlend;    // fade + glyph draws into trail buffer
static ComPtr<ID3D11BlendState>    g_opaqueBlend;         // final composite to backbuffer
static ComPtr<ID3D11Texture2D>     g_whiteTexture;        // 1x1 opaque white, used for the fade quad
static ComPtr<ID3D11ShaderResourceView> g_whiteSRV;

// D2D/DWrite are retained solely to rasterize the glyph atlases (one-time,
// at startup and on resize) via D3D11 interop; they never touch the
// per-frame render path.
static ComPtr<ID2D1Factory1>      g_d2dFactory;
static ComPtr<ID2D1Device>        g_d2dDevice;
static ComPtr<ID2D1DeviceContext> g_d2dContext;
static ComPtr<IDWriteFactory>     g_dwriteFactory;

// Per-instance data uploaded to the GPU for one glyph quad. destPos/destSize
// are in pixels; uv0/uv1 select the glyph's cell within the atlas strip;
// color is a premultiplied-alpha tint (brightnessMul baked in per layer).
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

// Dynamic instance buffers, one pair per layer (head/trail), resized as
// needed and refilled every frame. Kept as persistent D3D11 dynamic buffers
// so we're not allocating/destroying GPU resources every frame.
struct LayerInstanceBuffer {
    ComPtr<ID3D11Buffer> buffer;
    UINT capacity = 0; // in instances
};
static LayerInstanceBuffer g_headInstanceBuf[NUM_LAYERS];
static LayerInstanceBuffer g_trailInstanceBuf[NUM_LAYERS];

// CPU-side scratch, refilled each frame, reused across frames to avoid
// reallocating the vectors constantly.
static std::vector<GlyphInstance> g_headScratch[NUM_LAYERS];
static std::vector<GlyphInstance> g_trailScratch[NUM_LAYERS];

static bool g_isPreview = false;
static ULONGLONG g_startTick = 0;
static const ULONGLONG STARTUP_GRACE_MS = 1000;
static bool InGracePeriod() { return (GetTickCount64() - g_startTick) < STARTUP_GRACE_MS; }

static POINT g_lastMousePos{};
static bool  g_mouseInit = false;
static bool  g_isVisible = true; // false while minimized/fully occluded

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

// Above this canvas area (roughly a single 1080p display), column count grows
// linearly with area even though a human's perceived "density" only cares
// about glyphs-per-visible-area. On large single displays, 4K panels, or
// triple-monitor spans, that means far more live columns/symbols than the
// effect was ever tuned for. We scale the density down above the threshold
// so glyphs-per-area stays roughly constant instead of raw column count
// growing unbounded. Below the threshold this is a no-op (scale == 1.0),
// so normal single-monitor setups render identically to before.
static const float DENSITY_BASELINE_AREA = 1920.0f * 1080.0f;

static float ComputeDensityScale(int width, int height) {
    float area = static_cast<float>(width) * static_cast<float>(height);
    if (area <= DENSITY_BASELINE_AREA) return 1.0f;
    float scale = DENSITY_BASELINE_AREA / area;
    // Floor it so extreme spans (e.g. 3x 4K) still keep a reasonable amount
    // of visible rain rather than thinning out too aggressively.
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

    // Rebuild the per-layer index buckets once, right after g_columns is
    // finalized, so DrawFrame never has to filter col.layer on every frame.
    g_columnsByLayer.assign(NUM_LAYERS, {});
    for (size_t i = 0; i < g_columns.size(); ++i) {
        g_columnsByLayer[g_columns[i].layer].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// Shader Source (compiled once at startup)
// ---------------------------------------------------------------------------
// Single shader pair handles all three draw kinds (fade quad, glyph
// instances, final composite) by varying the bound texture and the
// per-instance color tint. destPos/destSize are in pixels; the vertex
// shader converts to NDC using the viewport-size constant buffer.
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
// Direct3D11 Setup Execution Contracts
// ---------------------------------------------------------------------------
static HRESULT CreateD3DDeviceAndSwapChain(HWND hwnd, int width, int height) {
    DXGI_SWAP_CHAIN_DESC scd = {};
    // DISCARD (the legacy bit-block-transfer swap effect) is only a valid
    // combination with a single back buffer - passing BufferCount=2 here
    // causes CreateSwapChain to return E_INVALIDARG (0x80070057) on
    // present-day WDDM drivers. FLIP_* effects support BufferCount>1, but
    // DISCARD does not; since this is a screensaver (not latency/perf
    // critical) a single back buffer is fine.
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
    // DISCARD is the broadest-compatible swap effect; this is a screensaver,
    // not a latency-critical app, so we don't need FLIP_SEQUENTIAL's extra
    // bookkeeping.
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL chosen{};

    // BGRA_SUPPORT is required so the same textures can be interop'd into
    // Direct2D for the one-time atlas rasterization pass.
    UINT deviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    UINT debugFlag = 0;
#ifdef _DEBUG
    debugFlag = D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags | debugFlag,
        levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
        &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);

    // D3D11_CREATE_DEVICE_DEBUG requires the optional "Graphics Tools"
    // Windows feature (the D3D SDK debug layer). On a machine that doesn't
    // have it installed, requesting this flag makes the call fail with
    // E_INVALIDARG (0x80070057) - not a hardware/feature-level problem at
    // all. Retry once without it before assuming the driver itself is at
    // fault.
    if (FAILED(hr) && debugFlag != 0) {
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlags,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
    }

    // Hardware driver may be missing/unable to satisfy BGRA_SUPPORT at any
    // of the requested feature levels (common under RDP/some VMs/old GPUs).
    // Fall back to the WARP software rasterizer rather than failing init
    // outright - this is a screensaver, so WARP's performance is acceptable.
    if (FAILED(hr)) {
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, deviceFlags,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &scd, &g_swapChain, &g_d3dDevice, &chosen, &g_d3dContext);
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

    // Static unit quad (triangle strip: 0,0 / 1,0 / 0,1 / 1,1), scaled and
    // positioned per-instance in the vertex shader.
    const float quadVerts[] = { 0,0, 1,0, 0,1, 1,1 };
    D3D11_BUFFER_DESC qbd = {};
    qbd.ByteWidth = sizeof(quadVerts);
    qbd.Usage = D3D11_USAGE_IMMUTABLE;
    qbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA qinit = { quadVerts, 0, 0 };
    hr = g_d3dDevice->CreateBuffer(&qbd, &qinit, &g_quadVertexBuffer);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBuffer(quadVertexBuffer)"; return hr; }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(float) * 4; // float2 viewport + float2 padding, 16-byte aligned
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

    // Premultiplied-alpha blend: used for the fade quad and glyph draws into
    // the persistent trail buffer, since the atlas textures come out of
    // Direct2D premultiplied.
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

    // Opaque (blend-disabled) state for the final trail->backbuffer
    // composite, matching the original's D2D1_ALPHA_MODE_IGNORE behavior:
    // a straight overwrite regardless of the trail buffer's alpha channel.
    D3D11_BLEND_DESC obd = {};
    obd.RenderTarget[0].BlendEnable = FALSE;
    obd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = g_d3dDevice->CreateBlendState(&obd, &g_opaqueBlend);
    if (FAILED(hr)) { if (failedCall) *failedCall = L"CreateBlendState(opaque)"; return hr; }

    // 1x1 opaque white texture, sampled by the fade quad; tinted by its
    // instance color (0,0,0,0.16) to produce the translucent black fade.
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

    // Persistent trail buffer: same size as the client area, never cleared
    // after this point (only faded), exactly like the old g_trailTarget.
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
    return S_OK;
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

    for (int i = 0; i < NUM_LAYERS; ++i) {
        hr = BuildAtlasForLayer(i);
        if (FAILED(hr)) { if (failedStage) *failedStage = L"BuildAtlasForLayer"; return hr; }
    }
    return S_OK;
}

static HRESULT EnsureInstanceCapacity(LayerInstanceBuffer& lib, UINT needed) {
    if (needed <= lib.capacity && lib.buffer) return S_OK;
    // Grow with slack so we're not reallocating every time the visible
    // glyph count fluctuates by one.
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
static void DrawFrame(DWORD tickCount) {
    if (!g_d3dContext || !g_trailRTV || !g_backBufferRTV) return;

    // --- Simulation + instance-list build (unchanged math, new sink) -------
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
                // Precomputed-tick re-roll: avoids a modulo per symbol per
                // frame. Only recomputes the next trigger tick when it
                // actually fires, instead of testing (tickCount % interval)
                // every single frame for every symbol.
                if (s.interval > 0 && tickCount >= s.nextChangeTick) {
                    s.value = RandomKatakana();
                    s.nextChangeTick = tickCount + static_cast<DWORD>(s.interval);
                }

                s.y += static_cast<float>(s.speed) * cfg.speedMul * 0.6f;
                if (s.y > g_height) s.y = -static_cast<float>(cfg.fontSize);

                // Skip glyphs currently outside the visible viewport (e.g.
                // trailing symbols still above frame, or mid-wrap). Position
                // still updates above so parallax speed, wrapping, and depth
                // ordering are unaffected - this only avoids adding an
                // instance for something that would render nothing.
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
                // Premultiplied tint: atlas rgb is already baked with the
                // layer's head/trail color at full alpha, so scaling all
                // four channels by brightnessMul reduces both opacity and
                // premultiplied color together, matching the original
                // DrawBitmap(..., opacity, ...) call exactly.
                inst.colorR = inst.colorG = inst.colorB = inst.colorA = brightness;

                if (s.isHead) g_headScratch[layer].push_back(inst);
                else          g_trailScratch[layer].push_back(inst);
            }
        }
    }

    // --- Upload viewport constant buffer (shared by every draw this frame) -
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(g_d3dContext->Map(g_viewportCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        float vp[4] = { static_cast<float>(g_width), static_cast<float>(g_height), 0.0f, 0.0f };
        memcpy(mapped.pData, vp, sizeof(vp));
        g_d3dContext->Unmap(g_viewportCB.Get(), 0);
    }

    // --- Common pipeline state, shared by all draws this frame -------------
    UINT stride = sizeof(float) * 2, offset = 0;
    g_d3dContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    g_d3dContext->IASetInputLayout(g_inputLayout.Get());
    g_d3dContext->VSSetShader(g_vertexShader.Get(), nullptr, 0);
    g_d3dContext->VSSetConstantBuffers(0, 1, g_viewportCB.GetAddressOf());
    g_d3dContext->PSSetShader(g_pixelShader.Get(), nullptr, 0);
    g_d3dContext->PSSetSamplers(0, 1, g_samplerState.GetAddressOf());

    D3D11_VIEWPORT vp = { 0, 0, static_cast<float>(g_width), static_cast<float>(g_height), 0.0f, 1.0f };
    g_d3dContext->RSSetViewports(1, &vp);

    // --- Pass 1: fade + glyphs into the persistent trail buffer -------------
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

    // --- Pass 2: composite trail buffer onto the backbuffer (opaque) -------
    ID3D11ShaderResourceView* nullSRV = nullptr;
    g_d3dContext->PSSetShaderResources(0, 1, &nullSRV); // unbind before rebinding trail texture as RTV->SRV
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

    // Sync-interval 1 paces presentation to vsync (same intent as the
    // earlier D2D1_PRESENT_OPTIONS_NONE change) rather than flipping as
    // fast as possible.
    HRESULT hr = g_swapChain->Present(1, 0);

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

        if (!g_isPreview) ShowCursor(FALSE);
        ResetMouseTracking(hwnd);
        SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL_MS, nullptr);
        return 0;
    }
    case WM_TIMER: {
        // Skip rendering entirely while minimized or fully occluded - the
        // animation state (positions, glyphs) simply doesn't advance during
        // that time, same as if the timer had never fired. No frames are
        // dropped or skipped while actually on screen.
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

            // Nothing to do if the size didn't actually change (e.g. a
            // restore-from-minimize that lands back at the same client
            // rect) - avoids tearing down/rebuilding the swap chain buffers
            // for free.
            if (newWidth != g_width || newHeight != g_height) {
                g_width = newWidth;
                g_height = newHeight;

                // Release everything that holds a reference to the swap
                // chain's back buffer or is sized off the old client area
                // before calling ResizeBuffers - D3D11 refuses to resize
                // while views onto the old buffers are still alive.
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
                    // Swap chain is in an unrecoverable state at this size -
                    // fall back to the same full rebuild path used for
                    // device-removed/reset, same as DrawFrame does.
                    DiscardDeviceResources();
                    if (SUCCEEDED(InitDirect2D(hwnd))) {
                        InitColumns(g_width, g_height);
                    }
                    return 0;
                }

                // The persistent trail buffer was just recreated at the new
                // size (and cleared to black in CreateSizeDependentResources),
                // so the rain layout is reseeded to match the new bounds -
                // same intent as the old g_trailTarget rebuild-and-clear.
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
    case WM_KEYDOWN:
        if (!g_isPreview && !InGracePeriod()) DestroyWindow(hwnd);
        return 0;

    case WM_ACTIVATE:
        if (!g_isPreview && !InGracePeriod() && LOWORD(wParam) == WA_INACTIVE) DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        KillTimer(hwnd, TIMER_ID);
        if (!g_isPreview) ShowCursor(TRUE);
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