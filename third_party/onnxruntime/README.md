# ONNX Runtime C API Header

This folder vendors the MIT-licensed ONNX Runtime 1.18.1 C API header so Vox Studio can
load `onnxruntime.dll` dynamically at runtime without linking against an import library.

Do not commit ONNX Runtime DLLs, provider DLLs, `.onnx` models, or user voice models here.
Installer packaging owns runtime delivery in a later pass.
