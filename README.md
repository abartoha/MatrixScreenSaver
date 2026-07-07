# Matrix Digital Rain Screensaver

![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D6?logo=windows&logoColor=white)
![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)
![Graphics API](https://img.shields.io/badge/graphics-Direct3D%2011%20%7C%20Direct2D-5C2D91?logo=direct3d&logoColor=white)
![License](https://img.shields.io/badge/license-Unlicensed-lightgrey)
![Build](https://img.shields.io/badge/build-Visual%20Studio%202019-purple?logo=visualstudio&logoColor=white)
![Status](https://img.shields.io/badge/status-active-brightgreen)

![Beware of the preview](https://raw.githubusercontent.com/abartoha/MatrixScreenSaver/refs/heads/master/output.gif?token=GHSAT0AAAAAAD63WXNV2RHJY2MAK5O35ZNA2SNRD3A)

A modern Windows screensaver inspired by the iconic **digital rain** from **The Matrix** film series. Built in modern C++ with Direct3D11 instanced rendering and Direct2D/DirectWrite glyph rasterization, it recreates the cascading streams of glowing katakana characters using a fully GPU-accelerated rendering pipeline designed for smooth, high-performance animation paced to your monitor's native refresh rate.

The primary goal of this project is to faithfully capture the atmosphere and aesthetic of the Matrix digital rain while remaining lightweight and entirely native to Windows.

An optional developer-oriented diagnostics overlay can be enabled for profiling and debugging the renderer. This mode displays real-time performance statistics and hardware utilization without affecting the core screensaver experience.

---

## Features

### Matrix Digital Rain

* Inspired by the iconic digital rain from *The Matrix*
* GPU-accelerated rendering using Direct3D11 instanced draw calls
* Multi-layer cascading character streams (4 parallax depth layers)
* Pre-rendered glyph atlas for efficient text rendering
* Multiple depth layers with independent:

  * Font sizes
  * Animation speeds
  * Brightness
  * Density
  * Color palettes
* Persistent fading trails rendered through an off-screen framebuffer
* Fast XorShift-based random number generator for glyph updates
* Native Win32 screensaver implementation (`.scr`)
* Render loop paced by the display's native refresh rate (e.g. 144Hz), driven by vsync rather than a fixed Win32 timer, with all animation using real elapsed time so speed stays consistent across frame rates

---

## Optional Debug & Performance Overlay

The project includes an optional benchmarking mode intended primarily for development, profiling, and performance diagnostics.

Benchmarking is controlled by a compile-time switch, `ENABLE_BENCHMARK_OVERLAY`, near the top of the source file:

```cpp
#define ENABLE_BENCHMARK_OVERLAY 1   // 0 disables and strips out all overlay/profiling code
```

When enabled, press **`B`** at runtime to toggle the overlay on/off without exiting the screensaver.

### Renderer Statistics

* Current FPS (rolling average, min/max, and p99 frame time)
* Total frames rendered / dropped presents
* Process CPU usage vs. total system CPU usage
* Process memory usage (working set / private bytes)

### System Monitoring

* Overall system memory usage
* Graphics adapter information (name, vendor)
* Dedicated vs. integrated GPU detection, with a note if a better discrete GPU was available but not selected
* Local VRAM usage / budget (where supported by the driver)

> **Note:** This overlay is a debugging feature intended for development and profiling. Simulated/estimated figures are labeled as such in the overlay; see [Notes](#notes) below.

---

## Graphics Pipeline

Rendering is entirely GPU accelerated and built on native Windows APIs:

* Direct3D11 (instanced rendering)
* Direct2D + DirectWrite (glyph atlas rasterization only, at startup/resize)
* DXGI
* Bitmap glyph atlases
* Off-screen trail rendering
* Vsync-paced presentation

Graphics resources are cached and reused to minimize allocations and maintain stable frame times.

---

## GPU Selection

The renderer automatically requests the preferred graphics adapter, prioritizing discrete GPUs over integrated devices whenever possible, and reports its choice in the debug overlay.

Selection order:

1. Explicit high-performance adapter request via `IDXGIFactory6::EnumAdapterByGpuPreference` (Windows 10 1803+)
2. `NvOptimusEnablement` / `AmdPowerXpressRequestHighPerformance` export hints (fallback for older systems)
3. Highest available dedicated VRAM among discrete adapters
4. Integrated GPU (fallback)

---

## Frame Rate & Performance Tuning

* `ENABLE_BENCHMARK_OVERLAY` — toggles the diagnostics overlay and all associated profiling code (see above).
* `UNCAP_FRAMERATE` — set to `1` to disable vsync (`Present(0, 0)`) and render as fast as the GPU allows, for measuring true max throughput. Defaults to `0` (vsync on), which paces the loop to your monitor's native refresh rate with no tearing and minimal wasted GPU work.

---

## Development

This project was developed using:

* **Microsoft Visual Studio 2019**
* **C++17**
* **Windows SDK**
* **Win32 API**
* **Direct3D11**
* **Direct2D**
* **DirectWrite**
* **DXGI 1.6**
* **Windows Performance Data Helper (PDH)**
* **Windows Runtime Library (WRL / Microsoft::WRL::ComPtr)**

### Dependencies

The project links against the following Windows libraries:

```text
d2d1.lib
dwrite.lib
d3d11.lib
dxgi.lib
d3dcompiler.lib
Pdh.lib
User32.lib
Gdi32.lib
Psapi.lib
```

No third-party libraries or frameworks are required. Everything is built using the native Windows graphics and performance APIs included with the Windows SDK.

---

## Compatibility

* Windows 10
* Windows 11

Requires a Direct3D11-capable graphics adapter. The renderer automatically prefers dedicated GPUs when multiple graphics adapters are present.

---

## Inspiration

This project is a recreation of the iconic **digital rain** seen throughout **The Matrix** film series.

Rather than reproducing the original effect exactly, the renderer focuses on capturing its distinctive visual style while taking advantage of modern GPU acceleration and native Windows rendering APIs. The result is a smooth, lightweight screensaver that remains faithful to the aesthetic while serving as an interesting graphics programming project.

---

## Notes

Local VRAM usage/budget is queried live via `IDXGIAdapter3::QueryVideoMemoryInfo` where the driver supports it; the overlay reports "unsupported" rather than a fabricated number when it doesn't. CPU/GPU temperatures and voltage readings are **not** currently implemented — Windows does not expose these sensors through standard user-mode APIs, and accurate readings would require vendor-specific SDKs or third-party hardware monitoring libraries such as LibreHardwareMonitor or HWiNFO.

---

## Changelog

All notable changes to this project are documented below.

### [Unreleased]

#### Added
- Compile-time `ENABLE_BENCHMARK_OVERLAY` macro to fully include or strip out the diagnostics overlay and all profiling code at build time.
- Compile-time `UNCAP_FRAMERATE` macro to optionally disable vsync for max-throughput benchmarking.
- Runtime **B** key toggle for the diagnostics overlay.
- Live FPS, frame-time (min/avg/max/p99), total frames rendered, and dropped-present counters.
- Process vs. system CPU usage via PDH.
- Process (working set / private bytes) and system memory usage.
- GPU adapter enumeration and reporting, including discrete-vs-integrated detection and a note when a better discrete GPU is present but unused.
- Local VRAM usage/budget query via `IDXGIAdapter3::QueryVideoMemoryInfo`.
- Explicit high-performance GPU adapter request via `IDXGIFactory6::EnumAdapterByGpuPreference`, alongside the existing `NvOptimusEnablement` / `AmdPowerXpressRequestHighPerformance` export hints.

#### Changed
- Render loop is no longer paced by a fixed 16ms Win32 timer (`WM_TIMER`). Fullscreen rendering is now driven by a `PeekMessage` loop in `wWinMain`, paced by `Present(1, 0)`'s vsync wait, allowing the renderer to reach the display's full native refresh rate (previously capped around ~40 FPS on high-refresh displays regardless of GPU headroom).
- Symbol fall-speed animation now scales by measured real elapsed time (delta time) instead of assuming a fixed ~60 FPS tick, so animation speed stays consistent whether running at 60Hz, 144Hz, or uncapped.
- DXGI header dependency bumped from `dxgi1_2.h` to `dxgi1_6.h` to support `IDXGIFactory6` / `IDXGIAdapter3`.

#### Fixed
- Frame rate no longer silently plateaus far below the monitor's refresh rate on capable hardware.

---
