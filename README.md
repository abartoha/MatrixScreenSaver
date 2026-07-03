# Matrix Digital Rain Screensaver

A modern Windows screensaver inspired by the iconic **digital rain** from **The Matrix** film series. Built in modern C++ with Direct2D and DirectWrite, it recreates the cascading streams of glowing katakana characters using a fully GPU-accelerated rendering pipeline designed for smooth, high-performance animation.

The primary goal of this project is to faithfully capture the atmosphere and aesthetic of the Matrix digital rain while remaining lightweight and entirely native to Windows.

An optional developer-oriented diagnostics overlay can be enabled for profiling and debugging the renderer. This mode displays real-time performance statistics and hardware utilization without affecting the core screensaver experience.

---

## Features

### Matrix Digital Rain

* Inspired by the iconic digital rain from *The Matrix*
* GPU-accelerated rendering using Direct2D
* Multi-layer cascading character streams
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

---

## Optional Debug & Performance Overlay

The project includes an optional benchmarking mode intended primarily for development, profiling, and performance diagnostics.

Launching the screensaver with the `/b` argument enables a real-time overlay displaying renderer statistics and selected Windows performance counters.

### Renderer Statistics

* Current FPS
* Process CPU usage
* Process memory usage
* Local VRAM allocation

### System Monitoring

* Overall CPU utilization
* GPU utilization
* Physical memory usage
* Graphics adapter information
* Dedicated vs. integrated GPU detection
* Battery status (when available)

### Performance History

Live sparklines are rendered for:

* FPS
* CPU usage
* GPU usage
* RAM usage

> **Note:** This overlay is a debugging feature and is disabled during normal screensaver operation.

---

## Graphics Pipeline

Rendering is entirely GPU accelerated and built on native Windows APIs:

* Direct2D
* DirectWrite
* DXGI
* Bitmap glyph atlases
* Off-screen trail rendering
* Immediate presentation

Graphics resources are cached and reused to minimize allocations and maintain stable frame times.

---

## GPU Selection

The renderer automatically selects the preferred graphics adapter, prioritizing discrete GPUs over integrated devices whenever possible.

Selection order:

1. Dedicated GPU
2. Highest available dedicated VRAM
3. Integrated GPU (fallback)

---

## Benchmark Mode

Enable the diagnostics overlay by launching:

```text
MatrixScreensaver.scr /b
```
## Development

This project was developed using:

* **Microsoft Visual Studio 2019**
* **C++17**
* **Windows SDK**
* **Win32 API**
* **Direct2D**
* **DirectWrite**
* **DXGI 1.4**
* **Windows Performance Data Helper (PDH)**
* **Windows Runtime Library (WRL / Microsoft::WRL::ComPtr)**

### Dependencies

The project links against the following Windows libraries:

```text
d2d1.lib
dwrite.lib
dxgi.lib
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

Requires a Direct2D-capable graphics adapter. The renderer automatically prefers dedicated GPUs when multiple graphics adapters are present.

---

## Inspiration

This project is a recreation of the iconic **digital rain** seen throughout **The Matrix** film series.

Rather than reproducing the original effect exactly, the renderer focuses on capturing its distinctive visual style while taking advantage of modern GPU acceleration and native Windows rendering APIs. The result is a smooth, lightweight screensaver that remains faithful to the aesthetic while serving as an interesting graphics programming project.


---

## Notes

CPU temperatures, GPU temperatures, and voltage readings shown by the debug overlay are **simulated** values intended only to demonstrate the interface.

Windows does not expose these sensors through standard user-mode APIs. Accurate readings would require vendor-specific SDKs or third-party hardware monitoring libraries such as LibreHardwareMonitor or HWiNFO.
