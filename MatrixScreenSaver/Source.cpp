// MatrixScreensaver.cpp
// A GPU-accelerated Windows screensaver (.scr) implementing a falling-katakana
// "Matrix rain" effect using Direct2D, with multiple parallax depth layers.

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
#include <dwrite.h>
#include <wrl/client.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
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
// Direct2D / Pipeline Graphics Core
// ---------------------------------------------------------------------------
static ComPtr<ID2D1Factory>          g_d2dFactory;
static ComPtr<ID2D1HwndRenderTarget> g_renderTarget;
static ComPtr<IDWriteFactory>        g_dwriteFactory;
static ComPtr<ID2D1SolidColorBrush>  g_fadeBrush;
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
                s.interval = 5 + static_cast<int>(FastRandBounded(25));
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
        // D2D1_PRESENT_OPTIONS_NONE lets DWM pace presentation to vsync instead
        // of flipping as fast as possible. On high-refresh displays this cuts
        // GPU/composite work substantially; the 16ms timer already caps our
        // intended frame rate near 60fps, so there's no visible change to the
        // rain's motion, only less wasted presentation work between frames.
        D2D1::HwndRenderTargetProperties(hwnd, size, D2D1_PRESENT_OPTIONS_NONE),
        &g_renderTarget);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 0.16f), &g_fadeBrush);
    if (FAILED(hr)) return hr;

    hr = g_renderTarget->CreateCompatibleRenderTarget(&g_trailTarget);
    if (FAILED(hr)) return hr;

    g_trailTarget->BeginDraw();
    g_trailTarget->Clear(D2D1::ColorF(D2D1::ColorF::Black));
    g_trailTarget->EndDraw();

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

    for (int i = 0; i < NUM_LAYERS; ++i) {
        hr = BuildAtlasForLayer(i);
        if (FAILED(hr)) return hr;
    }
    return S_OK;
}

// ---------------------------------------------------------------------------
// Frame Rendering
// ---------------------------------------------------------------------------
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

                // Skip the draw call entirely for glyphs currently outside
                // the visible viewport (e.g. trailing symbols still above
                // frame, or a symbol mid-wrap). Position/state still update
                // above so parallax speed, wrapping, and depth ordering are
                // completely unaffected - this only avoids issuing a
                // DrawBitmap call for something that would render nothing.
                if (s.y + cellF < 0.0f || s.y > static_cast<float>(g_height)) continue;

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
    HRESULT hr = g_renderTarget->EndDraw();

    if (hr == D2DERR_RECREATE_TARGET) {
        DiscardDeviceResources();
        if (SUCCEEDED(CreateDeviceResources(g_hwnd))) {
            for (int i = 0; i < NUM_LAYERS; ++i) BuildAtlasForLayer(i);
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

        if (FAILED(InitDirect2D(hwnd))) {
            MessageBoxW(hwnd, L"Direct2D initialization failed.", L"Matrix Screensaver", MB_OK | MB_ICONERROR);
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
        if (g_renderTarget) {
            RECT rc;
            GetClientRect(hwnd, &rc);
            g_width = rc.right - rc.left;
            g_height = rc.bottom - rc.top;
            D2D1_SIZE_U size = D2D1::SizeU(g_width, g_height);
            g_renderTarget->Resize(size);

            // g_trailTarget (the offscreen bitmap render target holding the
            // fading trail buffer) must be rebuilt here at the new size and
            // the columns reseeded so the rain layout matches the new bounds.
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