# Native ONNX RVC

Status: Pass 9b native execution plumbing. The sidecar remains the production fallback until
the exported ONNX graph contract is validated against a reference RVC model and A/B output.

## Runtime Layout

Vox Studio probes ONNX Runtime dynamically from:

1. The application directory: `onnxruntime.dll`
2. `%LOCALAPPDATA%/VoxStudio/onnxruntime/onnxruntime.dll`
3. The normal Windows DLL search path

ONNX Runtime must be a dynamic MIT-licensed runtime. Do not commit the DLL to source control.
Installer packaging owns runtime delivery in the installer pass. The source tree vendors only
the official MIT C API header under `third_party/onnxruntime/include/` so the app can load the
DLL dynamically without an import library.

## Native Model Layout

Native model bundles live under:

`%LOCALAPPDATA%/VoxStudio/rvc_onnx_models/<model_id>/`

Each bundle contains:

```json
{
  "model_id": "hero",
  "generator_onnx": "generator.onnx",
  "hubert_onnx": "hubert.onnx",
  "f0_onnx": "rmvpe.onnx",
  "sample_rate": 48000,
  "hop_length": 160,
  "graph_contract": {
    "required": false,
    "generator": {
      "inputs": [
        {
          "name": "phone",
          "element_type": "float32",
          "dimensions": [1, -1, 256],
          "role": "content"
        },
        {"name": "pitchf", "element_type": "float32", "dimensions": [1, -1], "role": "f0"}
      ],
      "outputs": [
        {"name": "wave", "element_type": "float32", "dimensions": [1, -1], "role": "audio"}
      ]
    },
    "hubert": {
      "inputs": [
        {"name": "waveform", "element_type": "float32", "dimensions": [1, -1], "role": "audio"}
      ],
      "outputs": [
        {
          "name": "features",
          "element_type": "float32",
          "dimensions": [1, -1, 256],
          "role": "content"
        }
      ]
    },
    "f0": {
      "inputs": [
        {"name": "waveform", "element_type": "float32", "dimensions": [1, -1], "role": "audio"}
      ],
      "outputs": [
        {"name": "pitchf", "element_type": "float32", "dimensions": [1, -1], "role": "f0"}
      ]
    }
  }
}
```

The `.onnx` files are user/runtime artifacts and must not be committed. Contract dimensions
use `-1` as a wildcard for dynamic axes. Set `graph_contract.required` to `false` while
inspecting a new export, then turn it on once the generator, HuBERT, and RMVPE graph I/O
names and tensor contracts are confirmed.

## Semantic Tensor Roles

`inspect_onnx_contract.py` emits each discovered tensor with an empty `role` field. Fill those
roles before attempting native conversion:

| Role | Required location | Meaning |
|---|---|---|
| `audio` | HuBERT input, F0 input, generator output | PCM waveform tensor |
| `content` | HuBERT output, generator input | Content/features tensor |
| `f0` | F0 output, generator input | Pitch contour tensor |
| `speaker` | Optional generator input | Speaker id tensor |
| `length` | Optional generator input | Sequence/audio length tensor |
| `noise` | Optional generator input | Stochastic generator input |

The C++ engine resolves a pipeline plan from these roles. Missing or duplicate semantic roles
fail before the native path can be used.

## Input Tensor Preparation

`NativeRvcAudio` converts a live `OnnxRvcRequest` from little-endian PCM16 into mono float
tensors before native execution:

1. Decode PCM16 using the request sample rate and channel count.
2. Mix multichannel input to mono.
3. Resample HuBERT input to 16 kHz and F0 input to the model bundle sample rate.
4. Replace the single dynamic axis in each audio tensor shape with the prepared sample count.

Fixed-shape audio tensors are rejected unless the prepared sample count exactly matches the
declared shape. This keeps the native path from entering ORT execution with malformed live mic
chunks.

## ORT Stage Execution

Once the bundle is configured and the pipeline plan resolves, `OnnxRvcEngine::convertChunk`
executes the native stages in order:

1. Run HuBERT with the prepared 16 kHz audio tensor and read the `content` output.
2. Run RMVPE/F0 with the model-rate audio tensor and read the `f0` output.
3. Create default generator auxiliary tensors for supported roles:
   - `speaker`: int64 zeros.
   - `length`: int64 content-frame count.
   - `noise`: float32 zeros with dynamic axes resolved to the content-frame count.
4. Run the generator with content, F0, and supported auxiliaries.
5. Convert the float32 generator `audio` output to PCM16.

Unsupported auxiliary roles fail fast with a clear error so model-specific contracts can be
extended deliberately. Real native quality/latency validation still requires a reference ONNX
bundle and sidecar output for the A/B gate.

Generate a contract from real ONNX files with:

```powershell
python tools\scripts\inspect_onnx_contract.py `
  --generator C:\models\hero\generator.onnx `
  --hubert C:\models\hero\hubert.onnx `
  --f0 C:\models\hero\rmvpe.onnx `
  --required `
  --out C:\models\hero\graph_contract.json
```

Then assemble a native bundle from externally exported ONNX artifacts:

```powershell
python tools\scripts\export_rvc_to_onnx.py `
  --model-id hero `
  --pth C:\models\hero\source_checkpoint.pth `
  --generator-onnx C:\models\hero_exports\generator.onnx `
  --hubert-onnx C:\models\hero_exports\hubert.onnx `
  --rmvpe-onnx C:\models\hero_exports\rmvpe.onnx `
  --graph-contract C:\models\hero\graph_contract.json `
  --require-contract `
  --out %LOCALAPPDATA%\VoxStudio\rvc_onnx_models\hero
```

The script copies those external ONNX files into the runtime bundle layout and writes
`native_manifest.json`. The `.pth` and `.onnx` files remain runtime/user artifacts and must not
be committed to source control.

## A/B Spectral Comparison

Render the native output for a known input WAV with:

```powershell
C:\_b\voxstudio\r\Release\VoxStudioRvcNativeSmoke.exe `
  --runtime C:\VoxStudioRuntime\onnxruntime.dll `
  --bundle C:\models\hero `
  --input C:\renders\hero_input.wav `
  --output C:\renders\hero_native.wav `
  --report C:\renders\hero_native_report.json
```

Then render the same input through the sidecar path and compare the resulting PCM WAV files with:

```powershell
python tools\scripts\compare_rvc_outputs.py `
  --sidecar C:\renders\hero_sidecar.wav `
  --native C:\renders\hero_native.wav `
  --max-spectral-db 2.0 `
  --out C:\renders\hero_ab_report.json
```

The tool converts multichannel 16-bit PCM WAV input to mono, estimates native-vs-sidecar lag,
then compares aligned Hann-windowed spectra. It exits with `0` when
`mean_abs_spectral_db <= max_spectral_db` and `2` when the render fails the threshold. Positive
`native_lag_samples` means the native render starts later than the sidecar render.

## Pass 9b Gate

Once native render, sidecar render, and A/B reports exist, validate the full Pass 9b evidence
with:

```powershell
python tools\scripts\validate_rvc_pass9b_gate.py `
  --ab-report C:\renders\hero_ab_report.json `
  --native-report C:\renders\hero_native_report.json `
  --sidecar-report C:\renders\hero_sidecar_report.json `
  --sidecar-footprint %LOCALAPPDATA%\VoxStudio\rvc_sidecar `
  --native-footprint %LOCALAPPDATA%\VoxStudio\onnxruntime `
  --native-footprint %LOCALAPPDATA%\VoxStudio\rvc_onnx_models\hero `
  --min-size-reduction-mb 300 `
  --out C:\renders\hero_pass9b_gate.json
```

The gate passes only when spectral error is within threshold, native latency is no worse than
sidecar latency, and the measured native footprint is at least 300 MB smaller than the sidecar
footprint.

## Compatibility Matrix

| Component | Required | Notes |
|---|---:|---|
| ONNX Runtime | 1.18+ | Dynamic `onnxruntime.dll`, CPU provider first |
| CUDA EP | Optional | Required for the <=300 ms GPU target |
| Generator | RVC v2 ONNX | Input/output names still under validation |
| HuBERT encoder | ONNX | 16 kHz content features |
| F0 extractor | RMVPE ONNX | Preferred over Crepe for Pass 9b |

## Validation Gates

Native mode is selectable in Settings. When a user selects Native, Vox Studio now requires
`onnxruntime.dll` to load, requires ORT sessions to open for the generator, HuBERT, and RMVPE
graphs, captures each graph's input/output tensor names, element types, and shapes, and checks
the manifest `graph_contract` when present. The Live Mic panel can drive native conversion for
validation when a native bundle is installed under
`%LOCALAPPDATA%/VoxStudio/rvc_onnx_models/<model_id>/`. The sidecar remains the production
fallback until:

1. A reference RVC `.pth` is exported to the bundle layout.
2. The native engine runs generator, HuBERT, and RMVPE inference in process through Live Mic.
3. Output is A/B checked against the sidecar within +/-2 dB spectral error.
4. End-to-end latency is measured against the same audio and hardware.

## Source Notes

The ONNX Runtime C API exposes `OrtGetApiBase`, `GetApi`, and `GetVersionString`.
The native engine uses that ABI to probe the installed runtime and create ORT sessions without
statically linking an ONNX Runtime import library.
