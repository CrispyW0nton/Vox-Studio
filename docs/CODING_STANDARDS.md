# Vox Studio Coding Standards

## Language and Toolchain

- Use C++20 with MSVC 2022.
- CI treats warnings as errors with `/W4 /WX`.
- Use CMake 3.27 or newer and vcpkg manifest mode.
- Qt is LGPLv3 and must be dynamically linked.

## Ownership and Errors

- Use RAII for every owned resource.
- Do not use raw `new` or `delete`.
- Prefer `std::unique_ptr`; use `std::shared_ptr` only for real shared ownership.
- Fallible APIs return status/expected-style values. Do not throw exceptions across DLL boundaries.
- Mark status and owned-resource returns with `[[nodiscard]]`.

## Concurrency

- The UI thread runs only the Qt event loop.
- Audio callbacks are time-critical and must not lock or allocate.
- Cross-thread audio communication uses lock-free queues.
- Cross-thread Qt signals/slots must use explicit `Qt::QueuedConnection`.

## Style

- Types use `PascalCase`.
- Functions and variables use `camelCase`.
- Members use the `m_` prefix.
- Constants use a `k` prefix.
- Use `const` for non-mutating APIs and immutable values.
- Headers use `#pragma once`.
- Prefer forward declarations in headers where practical.
- Keep comments short and only where they clarify non-obvious intent.

## Logging and Secrets

- Use spdlog for logging.
- Do not use `std::cout` or `qDebug()` in release code.
- Do not commit API keys, `.env*` files, signing material, `.pth`, or `.onnx` files.
- ElevenLabs keys are read at runtime from DPAPI-backed storage.

## Tests

- Use Catch2 v3 for core unit tests.
- Use Qt Test for UI-level tests.
- Coverage target is at least 70% for `core`, `db`, `dsp`, and `net`.
