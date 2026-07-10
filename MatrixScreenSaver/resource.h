// resource.h
// Shared between the .rc script and the C++ source that loads the embedded
// font resource at runtime. Keep the ID and type string in sync in both
// places -- FindResource() must match exactly what the .rc file declares.
#pragma once

// Resource ID for the embedded Matrix replica TrueType font.
#define IDR_MATRIXFONT   101

// Custom resource type name for raw TTF binary blobs. This is not one of the
// Win32 predefined RT_* types (RT_FONT/RT_FONTDIR are for classic .FNT font
// *resources* baked into a module's font-directory table, not arbitrary TTF
// files) -- so we declare our own named type and treat the TTF as an opaque
// binary blob.
#define RT_MATRIXFONT    L"BINFONT"
