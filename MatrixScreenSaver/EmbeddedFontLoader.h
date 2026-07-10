// EmbeddedFontLoader.h
//
// Loads a TrueType font that has been embedded into this module's own PE
// resources (see Fonts.rc / resource.h) and registers it as a process-private
// font via AddFontMemResourceEx, so DirectWrite/GDI can resolve it by family
// name without the font ever being installed system-wide.
//
// Usage:
//   static EmbeddedFontLoader g_matrixFontLoader;
//   ...
//   g_matrixFontLoader.LoadFromResource(GetModuleHandle(nullptr),
//                                        IDR_MATRIXFONT, RT_MATRIXFONT);
//   ... later, on shutdown ...
//   g_matrixFontLoader.Unload();   // also happens automatically in dtor
//
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class EmbeddedFontLoader {
public:
    EmbeddedFontLoader() = default;
    ~EmbeddedFontLoader() { Unload(); }

    // Not copyable: this object owns a live AddFontMemResourceEx handle that
    // must be released exactly once via RemoveFontMemResourceEx.
    EmbeddedFontLoader(const EmbeddedFontLoader&) = delete;
    EmbeddedFontLoader& operator=(const EmbeddedFontLoader&) = delete;

    // Locates the named resource inside hModule's PE image, locks it into
    // memory, and registers it as a process-private font via
    // AddFontMemResourceEx. Returns true on success.
    //
    //   hModule      - module containing the embedded resource; pass
    //                   GetModuleHandle(nullptr) for the current EXE.
    //   resourceId   - e.g. IDR_MATRIXFONT
    //   resourceType - e.g. RT_MATRIXFONT (must match the .rc declaration)
    bool LoadFromResource(HMODULE hModule, int resourceId, const wchar_t* resourceType) {
        Unload(); // idempotent: release any previously installed font first

        HRSRC hResInfo = FindResourceW(hModule,
            MAKEINTRESOURCEW(resourceId), resourceType);
        if (!hResInfo) {
            LogFailure(L"FindResource", GetLastError());
            return false;
        }

        // LoadResource maps the resource data (no separate "unlock"/free is
        // needed for the HGLOBAL it returns -- it is not a moveable global
        // in the classic Win16 sense despite the legacy name).
        HGLOBAL hResData = LoadResource(hModule, hResInfo);
        if (!hResData) {
            LogFailure(L"LoadResource", GetLastError());
            return false;
        }

        DWORD fontSize = SizeofResource(hModule, hResInfo);
        if (fontSize == 0) {
            LogFailure(L"SizeofResource", GetLastError());
            return false;
        }

        // LockResource just returns a pointer into the module's already
        // mapped image; there is no matching UnlockResource call in Win32.
        void* pFontData = LockResource(hResData);
        if (!pFontData) {
            LogFailure(L"LockResource", GetLastError());
            return false;
        }

        DWORD numFontsInstalled = 0;
        m_fontHandle = AddFontMemResourceEx(
            pFontData,
            fontSize,
            nullptr,          // reserved, must be nullptr
            &numFontsInstalled);

        if (!m_fontHandle || numFontsInstalled == 0) {
            LogFailure(L"AddFontMemResourceEx", GetLastError());
            m_fontHandle = nullptr;
            return false;
        }

        // pFontData/hResData do not need to be retained: AddFontMemResourceEx
        // has already parsed and internalized whatever it needs into the GDI
        // font subsystem, and the resource-section memory backing pFontData
        // remains valid for the module's lifetime regardless.
        return true;
    }

    // Unregisters the font from this process. Safe to call multiple times
    // and safe to call even if LoadFromResource never succeeded.
    void Unload() {
        if (m_fontHandle) {
            RemoveFontMemResourceEx(m_fontHandle);
            m_fontHandle = nullptr;
        }
    }

    bool IsLoaded() const { return m_fontHandle != nullptr; }

private:
    static void LogFailure(const wchar_t* apiName, DWORD err) {
        wchar_t buf[256];
        swprintf_s(buf, L"[EmbeddedFontLoader] %s failed, GetLastError=%lu\n",
            apiName, err);
        OutputDebugStringW(buf);
    }

    HANDLE m_fontHandle = nullptr; // handle returned by AddFontMemResourceEx
};
