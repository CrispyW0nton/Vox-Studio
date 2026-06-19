#include "rvc/OnnxRvcEngine.h"

#include "audio/AudioFile.h"
#include "platform/win/AppPaths.h"
#include "rvc/NativeRvcAudio.h"

#include <Windows.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace voxstudio::rvc {
namespace {

constexpr auto kRuntimeDllName = L"onnxruntime.dll";
constexpr auto kManifestFileName = "native_manifest.json";
constexpr auto kRoleAudio = "audio";
constexpr auto kRoleContent = "content";
constexpr auto kRoleF0 = "f0";

using OrtGetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();

[[nodiscard]] core::Error onnxError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] std::string win32ErrorMessage(const DWORD errorCode) {
    if (errorCode == 0U) {
        return {};
    }
    return std::system_category().message(static_cast<int>(errorCode));
}

[[nodiscard]] bool fileExists(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error) && !error;
}

[[nodiscard]] std::filesystem::path siblingRuntimeDll() {
    std::array<wchar_t, MAX_PATH> buffer{};
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return {};
    }
    return std::filesystem::path{std::wstring{buffer.data(), length}}.parent_path() /
           kRuntimeDllName;
}

[[nodiscard]] std::filesystem::path configuredRuntimeDll() {
    const auto appData = platform::win::voxStudioDataPath();
    if (!appData.has_value()) {
        return {};
    }
    return *appData / "onnxruntime" / kRuntimeDllName;
}

[[nodiscard]] std::vector<std::filesystem::path> runtimeCandidatesFor(
    const std::filesystem::path& explicitRuntimeDllPath) {
    std::vector<std::filesystem::path> candidates;
    if (!explicitRuntimeDllPath.empty()) {
        candidates.push_back(explicitRuntimeDllPath);
    }
    const auto sibling = siblingRuntimeDll();
    if (!sibling.empty()) {
        candidates.push_back(sibling);
    }
    const auto configured = configuredRuntimeDll();
    if (!configured.empty()) {
        candidates.push_back(configured);
    }
    candidates.emplace_back(kRuntimeDllName);
    return candidates;
}

[[nodiscard]] core::Expected<bool> validateModelFile(const std::filesystem::path& path,
                                                     const std::string& label) {
    if (path.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument, label + " path is missing.");
    }
    if (path.extension() != ".onnx") {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               label + " must be an .onnx file.");
    }
    if (!fileExists(path)) {
        return onnxError(label + " file does not exist: " + path.string());
    }
    return true;
}

[[nodiscard]] core::Expected<bool> ortStatusToExpected(const OrtApi& api,
                                                       OrtStatus* status,
                                                       const std::string& context) {
    if (status == nullptr) {
        return true;
    }

    const char* message = api.GetErrorMessage(status);
    std::string detail = context;
    if (message != nullptr && message[0] != '\0') {
        detail += ": ";
        detail += message;
    }
    api.ReleaseStatus(status);
    return onnxError(detail);
}

[[nodiscard]] std::string tensorElementTypeName(
    const ONNXTensorElementDataType elementType) {
    switch (elementType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        return "float8e4m3fn";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        return "float8e4m3fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        return "float8e5m2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return "float8e5m2fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
        return "undefined";
    }
}

[[nodiscard]] std::string onnxTypeName(const ONNXType type) {
    switch (type) {
    case ONNX_TYPE_TENSOR:
        return "tensor";
    case ONNX_TYPE_SEQUENCE:
        return "sequence";
    case ONNX_TYPE_MAP:
        return "map";
    case ONNX_TYPE_OPAQUE:
        return "opaque";
    case ONNX_TYPE_SPARSETENSOR:
        return "sparse_tensor";
    case ONNX_TYPE_OPTIONAL:
        return "optional";
    case ONNX_TYPE_UNKNOWN:
    default:
        return "unknown";
    }
}

[[nodiscard]] std::vector<std::int64_t> parseDimensions(const nlohmann::json& json) {
    std::vector<std::int64_t> dimensions;
    if (!json.is_array()) {
        return dimensions;
    }
    dimensions.reserve(json.size());
    for (const auto& dimension : json) {
        dimensions.push_back(dimension.get<std::int64_t>());
    }
    return dimensions;
}

[[nodiscard]] std::vector<std::string> parseSymbolicDimensions(const nlohmann::json& json) {
    std::vector<std::string> dimensions;
    if (!json.is_array()) {
        return dimensions;
    }
    dimensions.reserve(json.size());
    for (const auto& dimension : json) {
        dimensions.push_back(dimension.get<std::string>());
    }
    return dimensions;
}

[[nodiscard]] OnnxTensorDescription parseTensorContract(const nlohmann::json& json) {
    OnnxTensorDescription tensor;
    tensor.name = json.value("name", std::string{});
    tensor.elementType = json.value("element_type", std::string{});
    if (json.contains("dimensions")) {
        tensor.dimensions = parseDimensions(json.at("dimensions"));
    }
    if (json.contains("symbolic_dimensions")) {
        tensor.symbolicDimensions = parseSymbolicDimensions(json.at("symbolic_dimensions"));
    }
    tensor.semanticRole = json.value("role", std::string{});
    return tensor;
}

[[nodiscard]] std::vector<OnnxTensorDescription> parseTensorContracts(
    const nlohmann::json& json) {
    std::vector<OnnxTensorDescription> tensors;
    if (!json.is_array()) {
        return tensors;
    }
    tensors.reserve(json.size());
    for (const auto& tensor : json) {
        tensors.push_back(parseTensorContract(tensor));
    }
    return tensors;
}

[[nodiscard]] OnnxSessionDescription parseSessionContract(const nlohmann::json& json,
                                                          std::string label) {
    OnnxSessionDescription session;
    session.label = std::move(label);
    if (json.contains("inputs")) {
        session.inputs = parseTensorContracts(json.at("inputs"));
    }
    if (json.contains("outputs")) {
        session.outputs = parseTensorContracts(json.at("outputs"));
    }
    return session;
}

[[nodiscard]] OnnxRvcGraphContract parseGraphContract(const nlohmann::json& json) {
    OnnxRvcGraphContract contract;
    if (!json.is_object()) {
        return contract;
    }
    contract.required = json.value("required", true);
    if (json.contains("generator")) {
        contract.generator = parseSessionContract(json.at("generator"), "RVC generator");
    }
    if (json.contains("hubert")) {
        contract.hubert = parseSessionContract(json.at("hubert"), "HuBERT encoder");
    }
    if (json.contains("f0")) {
        contract.f0 = parseSessionContract(json.at("f0"), "RMVPE F0 extractor");
    }
    return contract;
}

[[nodiscard]] bool sessionContractIsEmpty(const OnnxSessionDescription& session) {
    return session.inputs.empty() && session.outputs.empty();
}

[[nodiscard]] bool graphContractIsEmpty(const OnnxRvcGraphContract& contract) {
    return sessionContractIsEmpty(contract.generator) &&
           sessionContractIsEmpty(contract.hubert) && sessionContractIsEmpty(contract.f0);
}

[[nodiscard]] const OnnxTensorDescription* findTensorByName(
    const std::vector<OnnxTensorDescription>& tensors,
    const std::string& name) {
    const auto found = std::find_if(
        tensors.begin(),
        tensors.end(),
        [&name](const OnnxTensorDescription& tensor) { return tensor.name == name; });
    return found == tensors.end() ? nullptr : &(*found);
}

[[nodiscard]] core::Expected<bool> validateTensorContract(
    const OnnxTensorDescription& expected,
    const OnnxTensorDescription& actual,
    const std::string& context) {
    if (expected.name.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               context + " contract tensor name is missing.");
    }
    if (!expected.elementType.empty() && expected.elementType != actual.elementType) {
        return core::makeError(
            core::ErrorCode::InvalidArgument,
            context + " expected " + expected.name + " type " + expected.elementType +
                ", got " + actual.elementType + ".");
    }
    if (!expected.dimensions.empty()) {
        if (expected.dimensions.size() != actual.dimensions.size()) {
            return core::makeError(
                core::ErrorCode::InvalidArgument,
                context + " expected " + expected.name + " rank " +
                    std::to_string(expected.dimensions.size()) + ", got " +
                    std::to_string(actual.dimensions.size()) + ".");
        }
        for (size_t index = 0; index < expected.dimensions.size(); ++index) {
            const auto expectedDimension = expected.dimensions[index];
            if (expectedDimension >= 0 && expectedDimension != actual.dimensions[index]) {
                return core::makeError(
                    core::ErrorCode::InvalidArgument,
                    context + " expected " + expected.name + " dimension " +
                        std::to_string(index) + " to be " +
                        std::to_string(expectedDimension) + ", got " +
                        std::to_string(actual.dimensions[index]) + ".");
            }
        }
    }
    return true;
}

[[nodiscard]] core::Expected<bool> validateTensorListContract(
    const std::vector<OnnxTensorDescription>& expectedTensors,
    const std::vector<OnnxTensorDescription>& actualTensors,
    const std::string& context) {
    for (const auto& expected : expectedTensors) {
        const auto* actual = findTensorByName(actualTensors, expected.name);
        if (actual == nullptr) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   context + " missing tensor: " + expected.name + ".");
        }
        auto valid = validateTensorContract(expected, *actual, context);
        if (!valid) {
            return valid.error();
        }
    }
    return true;
}

[[nodiscard]] core::Expected<bool> validateSessionContract(
    const OnnxSessionDescription& expected,
    const OnnxSessionDescription& actual,
    const bool required) {
    if (sessionContractIsEmpty(expected)) {
        if (!required) {
            return true;
        }
        return core::makeError(core::ErrorCode::InvalidArgument,
                               expected.label + " graph contract is missing.");
    }

    auto inputsValid = validateTensorListContract(
        expected.inputs,
        actual.inputs,
        expected.label + " inputs");
    if (!inputsValid) {
        return inputsValid.error();
    }
    auto outputsValid = validateTensorListContract(
        expected.outputs,
        actual.outputs,
        expected.label + " outputs");
    if (!outputsValid) {
        return outputsValid.error();
    }
    return true;
}

[[nodiscard]] core::Expected<bool> validateGraphContractAgainstDescription(
    const OnnxRvcGraphContract& contract,
    const OnnxRvcModelDescription& description) {
    if (graphContractIsEmpty(contract)) {
        if (!contract.required) {
            return true;
        }
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Native RVC graph contract is missing.");
    }

    auto generatorValid =
        validateSessionContract(contract.generator, description.generator, contract.required);
    if (!generatorValid) {
        return generatorValid.error();
    }
    auto hubertValid =
        validateSessionContract(contract.hubert, description.hubert, contract.required);
    if (!hubertValid) {
        return hubertValid.error();
    }
    auto f0Valid = validateSessionContract(contract.f0, description.f0, contract.required);
    if (!f0Valid) {
        return f0Valid.error();
    }
    return true;
}

[[nodiscard]] std::string normalizedRole(std::string role) {
    std::ranges::transform(role, role.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return role;
}

[[nodiscard]] bool tensorHasRole(const OnnxTensorDescription& tensor,
                                 const std::string& role) {
    return normalizedRole(tensor.semanticRole) == role;
}

[[nodiscard]] core::Expected<OnnxRvcTensorBinding> requireTensorRole(
    const OnnxSessionDescription& session,
    const std::vector<OnnxTensorDescription>& tensors,
    const std::string& role,
    const std::string& direction) {
    std::vector<const OnnxTensorDescription*> matches;
    for (const auto& tensor : tensors) {
        if (tensorHasRole(tensor, role)) {
            matches.push_back(&tensor);
        }
    }

    if (matches.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               session.label + " " + direction +
                                   " contract needs tensor role '" + role + "'.");
    }
    if (matches.size() > 1U) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               session.label + " " + direction +
                                   " contract has duplicate tensor role '" + role + "'.");
    }
    return OnnxRvcTensorBinding{session.label, *matches.front()};
}

[[nodiscard]] std::vector<OnnxRvcTensorBinding> auxiliaryGeneratorInputs(
    const OnnxSessionDescription& generator) {
    std::vector<OnnxRvcTensorBinding> bindings;
    for (const auto& tensor : generator.inputs) {
        const auto role = normalizedRole(tensor.semanticRole);
        if (role.empty() || role == kRoleContent || role == kRoleF0) {
            continue;
        }
        bindings.push_back(OnnxRvcTensorBinding{generator.label, tensor});
    }
    return bindings;
}

[[nodiscard]] core::Expected<OnnxRvcPipelinePlan> resolvePipelinePlanFromContract(
    const OnnxRvcGraphContract& contract) {
    if (graphContractIsEmpty(contract)) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Native RVC graph contract is missing.");
    }

    auto hubertAudio =
        requireTensorRole(contract.hubert, contract.hubert.inputs, kRoleAudio, "input");
    if (!hubertAudio) {
        return hubertAudio.error();
    }
    auto hubertContent =
        requireTensorRole(contract.hubert, contract.hubert.outputs, kRoleContent, "output");
    if (!hubertContent) {
        return hubertContent.error();
    }
    auto f0Audio = requireTensorRole(contract.f0, contract.f0.inputs, kRoleAudio, "input");
    if (!f0Audio) {
        return f0Audio.error();
    }
    auto f0Pitch = requireTensorRole(contract.f0, contract.f0.outputs, kRoleF0, "output");
    if (!f0Pitch) {
        return f0Pitch.error();
    }
    auto generatorContent = requireTensorRole(
        contract.generator,
        contract.generator.inputs,
        kRoleContent,
        "input");
    if (!generatorContent) {
        return generatorContent.error();
    }
    auto generatorF0 =
        requireTensorRole(contract.generator, contract.generator.inputs, kRoleF0, "input");
    if (!generatorF0) {
        return generatorF0.error();
    }
    auto generatorAudio =
        requireTensorRole(contract.generator, contract.generator.outputs, kRoleAudio, "output");
    if (!generatorAudio) {
        return generatorAudio.error();
    }

    OnnxRvcPipelinePlan plan;
    plan.hubertAudioInput = std::move(hubertAudio.value());
    plan.hubertContentOutput = std::move(hubertContent.value());
    plan.f0AudioInput = std::move(f0Audio.value());
    plan.f0PitchOutput = std::move(f0Pitch.value());
    plan.generatorContentInput = std::move(generatorContent.value());
    plan.generatorF0Input = std::move(generatorF0.value());
    plan.generatorAudioOutput = std::move(generatorAudio.value());
    plan.generatorAuxiliaryInputs = auxiliaryGeneratorInputs(contract.generator);
    return plan;
}

class OrtRuntimeLibrary final {
public:
    OrtRuntimeLibrary(HMODULE library,
                      const OrtApi* api,
                      std::filesystem::path runtimePath,
                      std::string version)
        : m_library(library)
        , m_api(api)
        , m_info{true, std::move(runtimePath), std::move(version), "ONNX Runtime loaded."} {}

    ~OrtRuntimeLibrary() {
        if (m_library != nullptr) {
            FreeLibrary(m_library);
        }
    }

    OrtRuntimeLibrary(const OrtRuntimeLibrary&) = delete;
    OrtRuntimeLibrary& operator=(const OrtRuntimeLibrary&) = delete;
    OrtRuntimeLibrary(OrtRuntimeLibrary&&) = delete;
    OrtRuntimeLibrary& operator=(OrtRuntimeLibrary&&) = delete;

    [[nodiscard]] const OrtApi& api() const noexcept {
        return *m_api;
    }

    [[nodiscard]] const OnnxRuntimeInfo& info() const noexcept {
        return m_info;
    }

private:
    HMODULE m_library{nullptr};
    const OrtApi* m_api{nullptr};
    OnnxRuntimeInfo m_info;
};

struct OrtEnvDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtEnv* env) const noexcept {
        if (api != nullptr && env != nullptr) {
            api->ReleaseEnv(env);
        }
    }
};

struct OrtSessionOptionsDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtSessionOptions* options) const noexcept {
        if (api != nullptr && options != nullptr) {
            api->ReleaseSessionOptions(options);
        }
    }
};

struct OrtSessionDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtSession* session) const noexcept {
        if (api != nullptr && session != nullptr) {
            api->ReleaseSession(session);
        }
    }
};

struct OrtTypeInfoDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtTypeInfo* typeInfo) const noexcept {
        if (api != nullptr && typeInfo != nullptr) {
            api->ReleaseTypeInfo(typeInfo);
        }
    }
};

struct OrtTensorTypeAndShapeInfoDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtTensorTypeAndShapeInfo* info) const noexcept {
        if (api != nullptr && info != nullptr) {
            api->ReleaseTensorTypeAndShapeInfo(info);
        }
    }
};

struct OrtMemoryInfoDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtMemoryInfo* info) const noexcept {
        if (api != nullptr && info != nullptr) {
            api->ReleaseMemoryInfo(info);
        }
    }
};

struct OrtValueDeleter final {
    const OrtApi* api{nullptr};

    void operator()(OrtValue* value) const noexcept {
        if (api != nullptr && value != nullptr) {
            api->ReleaseValue(value);
        }
    }
};

using OrtEnvPtr = std::unique_ptr<OrtEnv, OrtEnvDeleter>;
using OrtSessionOptionsPtr = std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleter>;
using OrtSessionPtr = std::unique_ptr<OrtSession, OrtSessionDeleter>;
using OrtTypeInfoPtr = std::unique_ptr<OrtTypeInfo, OrtTypeInfoDeleter>;
using OrtTensorTypeAndShapeInfoPtr =
    std::unique_ptr<OrtTensorTypeAndShapeInfo, OrtTensorTypeAndShapeInfoDeleter>;
using OrtMemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleter>;
using OrtValuePtr = std::unique_ptr<OrtValue, OrtValueDeleter>;

struct OrtOwnedTensor final {
    std::vector<float> floatValues;
    std::vector<std::int64_t> int64Values;
    OrtValuePtr value;
};

struct OrtFloatTensor final {
    std::vector<std::int64_t> dimensions;
    std::vector<float> values;
};

struct NativeSessions final {
    OrtEnvPtr env;
    OrtSessionOptionsPtr options;
    OrtSessionPtr generator;
    OrtSessionPtr hubert;
    OrtSessionPtr f0;
    OnnxSessionDescription generatorDescription;
    OnnxSessionDescription hubertDescription;
    OnnxSessionDescription f0Description;
};

[[nodiscard]] std::size_t byteCountForFloats(const std::vector<float>& values) {
    return values.size() * sizeof(float);
}

[[nodiscard]] std::size_t byteCountForInt64(const std::vector<std::int64_t>& values) {
    return values.size() * sizeof(std::int64_t);
}

[[nodiscard]] core::Expected<OrtMemoryInfoPtr> createCpuMemoryInfo(const OrtApi& api) {
    OrtMemoryInfo* rawInfo = nullptr;
    auto created = ortStatusToExpected(
        api,
        api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &rawInfo),
        "Create ONNX CPU memory info");
    if (!created) {
        return created.error();
    }
    return OrtMemoryInfoPtr{rawInfo, OrtMemoryInfoDeleter{&api}};
}

[[nodiscard]] core::Expected<OrtOwnedTensor> createFloatTensor(
    const OrtApi& api,
    const OrtMemoryInfo* memoryInfo,
    const NativeRvcAudioTensor& tensor) {
    if (tensor.values.empty() || tensor.dimensions.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               tensor.name + " float tensor is empty.");
    }

    OrtOwnedTensor owned;
    owned.floatValues = tensor.values;
    OrtValue* rawValue = nullptr;
    auto created = ortStatusToExpected(
        api,
        api.CreateTensorWithDataAsOrtValue(memoryInfo,
                                           owned.floatValues.data(),
                                           byteCountForFloats(owned.floatValues),
                                           tensor.dimensions.data(),
                                           tensor.dimensions.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           &rawValue),
        "Create ONNX float tensor " + tensor.name);
    if (!created) {
        return created.error();
    }
    owned.value = OrtValuePtr{rawValue, OrtValueDeleter{&api}};
    return owned;
}

[[nodiscard]] core::Expected<OrtOwnedTensor> createAuxiliaryTensor(
    const OrtApi& api,
    const OrtMemoryInfo* memoryInfo,
    const NativeRvcAuxiliaryTensor& tensor) {
    if (tensor.dimensions.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               tensor.name + " auxiliary tensor shape is missing.");
    }

    OrtOwnedTensor owned;
    OrtValue* rawValue = nullptr;
    if (!tensor.floatValues.empty()) {
        owned.floatValues = tensor.floatValues;
        auto created = ortStatusToExpected(
            api,
            api.CreateTensorWithDataAsOrtValue(memoryInfo,
                                               owned.floatValues.data(),
                                               byteCountForFloats(owned.floatValues),
                                               tensor.dimensions.data(),
                                               tensor.dimensions.size(),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                               &rawValue),
            "Create ONNX auxiliary float tensor " + tensor.name);
        if (!created) {
            return created.error();
        }
    } else if (!tensor.int64Values.empty()) {
        owned.int64Values = tensor.int64Values;
        auto created = ortStatusToExpected(
            api,
            api.CreateTensorWithDataAsOrtValue(memoryInfo,
                                               owned.int64Values.data(),
                                               byteCountForInt64(owned.int64Values),
                                               tensor.dimensions.data(),
                                               tensor.dimensions.size(),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                               &rawValue),
            "Create ONNX auxiliary int64 tensor " + tensor.name);
        if (!created) {
            return created.error();
        }
    } else {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               tensor.name + " auxiliary tensor values are missing.");
    }

    owned.value = OrtValuePtr{rawValue, OrtValueDeleter{&api}};
    return owned;
}

[[nodiscard]] core::Expected<std::vector<std::int64_t>> readOrtTensorDimensions(
    const OrtApi& api,
    const OrtValue* value,
    const std::string& context) {
    OrtTensorTypeAndShapeInfo* rawInfo = nullptr;
    auto typeLoaded = ortStatusToExpected(
        api,
        api.GetTensorTypeAndShape(value, &rawInfo),
        context + " type and shape");
    if (!typeLoaded) {
        return typeLoaded.error();
    }
    OrtTensorTypeAndShapeInfoPtr info{rawInfo, OrtTensorTypeAndShapeInfoDeleter{&api}};

    size_t dimensionCount = 0;
    auto countRead = ortStatusToExpected(
        api,
        api.GetDimensionsCount(info.get(), &dimensionCount),
        context + " dimension count");
    if (!countRead) {
        return countRead.error();
    }

    std::vector<std::int64_t> dimensions(dimensionCount);
    auto dimensionsRead = ortStatusToExpected(
        api,
        api.GetDimensions(info.get(), dimensions.data(), dimensions.size()),
        context + " dimensions");
    if (!dimensionsRead) {
        return dimensionsRead.error();
    }
    return dimensions;
}

[[nodiscard]] core::Expected<ONNXTensorElementDataType> readOrtTensorElementType(
    const OrtApi& api,
    const OrtValue* value,
    const std::string& context) {
    OrtTensorTypeAndShapeInfo* rawInfo = nullptr;
    auto typeLoaded = ortStatusToExpected(
        api,
        api.GetTensorTypeAndShape(value, &rawInfo),
        context + " type and shape");
    if (!typeLoaded) {
        return typeLoaded.error();
    }
    OrtTensorTypeAndShapeInfoPtr info{rawInfo, OrtTensorTypeAndShapeInfoDeleter{&api}};

    ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    auto elementTypeRead = ortStatusToExpected(
        api,
        api.GetTensorElementType(info.get(), &elementType),
        context + " element type");
    if (!elementTypeRead) {
        return elementTypeRead.error();
    }
    return elementType;
}

[[nodiscard]] core::Expected<std::size_t> elementCountFromDimensions(
    const std::vector<std::int64_t>& dimensions,
    const std::string& context) {
    if (dimensions.empty()) {
        return std::size_t{1};
    }

    std::size_t elementCount = 1;
    for (const auto dimension : dimensions) {
        if (dimension <= 0) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   context + " has unresolved output dimensions.");
        }
        if (elementCount > (std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(dimension))) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   context + " tensor is too large.");
        }
        elementCount *= static_cast<std::size_t>(dimension);
    }
    return elementCount;
}

[[nodiscard]] core::Expected<OrtFloatTensor> readFloatTensor(
    const OrtApi& api,
    OrtValue* value,
    const std::string& context) {
    auto elementType = readOrtTensorElementType(api, value, context);
    if (!elementType) {
        return elementType.error();
    }
    if (elementType.value() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               context + " must output float32.");
    }

    auto dimensions = readOrtTensorDimensions(api, value, context);
    if (!dimensions) {
        return dimensions.error();
    }
    auto elementCount = elementCountFromDimensions(dimensions.value(), context);
    if (!elementCount) {
        return elementCount.error();
    }

    void* rawData = nullptr;
    auto dataRead = ortStatusToExpected(
        api,
        api.GetTensorMutableData(value, &rawData),
        context + " data");
    if (!dataRead) {
        return dataRead.error();
    }
    const auto* first = static_cast<const float*>(rawData);
    OrtFloatTensor tensor;
    tensor.dimensions = std::move(dimensions.value());
    tensor.values.assign(first, first + elementCount.value());
    return tensor;
}

[[nodiscard]] core::Expected<OrtValuePtr> runSingleOutputSession(
    const OrtApi& api,
    OrtSession* session,
    const std::string& label,
    const std::vector<std::string>& inputNames,
    const std::vector<const OrtValue*>& inputs,
    const std::string& outputName) {
    if (inputNames.size() != inputs.size() || inputNames.empty() || outputName.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               label + " ONNX run binding is invalid.");
    }

    std::vector<const char*> inputNamePointers;
    inputNamePointers.reserve(inputNames.size());
    for (const auto& name : inputNames) {
        inputNamePointers.push_back(name.c_str());
    }
    const std::array<const char*, 1> outputNamePointers{outputName.c_str()};
    OrtValue* rawOutput = nullptr;
    auto ran = ortStatusToExpected(
        api,
        api.Run(session,
                nullptr,
                inputNamePointers.data(),
                inputs.data(),
                inputs.size(),
                outputNamePointers.data(),
                outputNamePointers.size(),
                &rawOutput),
        "Run " + label);
    if (!ran) {
        return ran.error();
    }
    return OrtValuePtr{rawOutput, OrtValueDeleter{&api}};
}

[[nodiscard]] core::Expected<std::int64_t> contentFrameCountFrom(
    const std::vector<std::int64_t>& dimensions) {
    if (dimensions.size() > 1U && dimensions[1] > 0) {
        return dimensions[1];
    }
    if (!dimensions.empty() && dimensions.back() > 0) {
        return dimensions.back();
    }
    return core::makeError(core::ErrorCode::InvalidArgument,
                           "HuBERT content output shape cannot provide frame count.");
}

[[nodiscard]] std::vector<std::uint8_t> pcmBytesFromFloatAudio(
    const std::vector<float>& samples,
    const int sampleRate) {
    audio::PcmAudioBuffer audio;
    audio.sampleRate = sampleRate;
    audio.channels = 1;
    audio.samples = samples;
    return audio::pcmToPcm16LittleEndian(audio);
}

} // namespace

class OnnxRvcEngine::Impl final {
public:
    explicit Impl(std::filesystem::path runtimeDllPath)
        : m_runtimeDllPath(std::move(runtimeDllPath)) {}

    [[nodiscard]] core::Expected<OnnxRuntimeInfo> probeRuntime() const {
        auto runtime = loadRuntime();
        if (runtime) {
            return runtime.value()->info();
        }
        if (runtime.error().code == core::ErrorCode::FileSystemFailure) {
            return OnnxRuntimeInfo{false, {}, {}, runtime.error().message};
        }
        return runtime.error();
    }

    [[nodiscard]] core::Expected<OnnxRvcModelBundle> loadModelBundle(
        const std::filesystem::path& bundleRoot) const {
        const auto manifestPath = bundleRoot / kManifestFileName;
        if (!fileExists(manifestPath)) {
            return onnxError("Native RVC manifest is missing: " + manifestPath.string());
        }

        try {
            std::ifstream input{manifestPath};
            const auto json = nlohmann::json::parse(input);

            OnnxRvcModelBundle bundle;
            bundle.rootPath = bundleRoot;
            bundle.modelId = json.at("model_id").get<std::string>();
            bundle.generatorModelPath =
                bundleRoot / json.at("generator_onnx").get<std::string>();
            bundle.hubertModelPath = bundleRoot / json.at("hubert_onnx").get<std::string>();
            bundle.f0ModelPath = bundleRoot / json.at("f0_onnx").get<std::string>();
            bundle.sampleRate = json.value("sample_rate", 48000);
            bundle.hopLength = json.value("hop_length", 160);
            if (json.contains("graph_contract")) {
                bundle.graphContract = parseGraphContract(json.at("graph_contract"));
            }

            auto validGenerator = validateModelFile(bundle.generatorModelPath, "RVC generator");
            if (!validGenerator) {
                return validGenerator.error();
            }
            auto validHubert = validateModelFile(bundle.hubertModelPath, "HuBERT encoder");
            if (!validHubert) {
                return validHubert.error();
            }
            auto validF0 = validateModelFile(bundle.f0ModelPath, "F0 extractor");
            if (!validF0) {
                return validF0.error();
            }
            if (bundle.modelId.empty() || bundle.sampleRate <= 0 || bundle.hopLength <= 0) {
                return core::makeError(core::ErrorCode::InvalidArgument,
                                       "Native RVC manifest metadata is invalid.");
            }

            return bundle;
        } catch (const std::exception& exception) {
            return onnxError(exception.what());
        }
    }

    [[nodiscard]] core::Expected<bool> configureModelBundle(OnnxRvcModelBundle bundle) {
        auto runtime = loadRuntime();
        if (!runtime) {
            return runtime.error();
        }

        auto validGenerator = validateModelFile(bundle.generatorModelPath, "RVC generator");
        if (!validGenerator) {
            return validGenerator.error();
        }
        auto validHubert = validateModelFile(bundle.hubertModelPath, "HuBERT encoder");
        if (!validHubert) {
            return validHubert.error();
        }
        auto validF0 = validateModelFile(bundle.f0ModelPath, "F0 extractor");
        if (!validF0) {
            return validF0.error();
        }

        auto sessions = createSessions(*runtime.value(), bundle);
        if (!sessions) {
            return sessions.error();
        }

        m_runtimeInfo = runtime.value()->info();
        m_runtime = std::move(runtime.value());
        m_sessions = std::move(sessions.value());
        m_modelBundle = std::move(bundle);
        return true;
    }

    [[nodiscard]] core::Expected<OnnxRvcModelDescription> describeConfiguredModel() const {
        if (m_modelBundle.modelId.empty() || m_sessions == nullptr) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC model bundle is not configured.");
        }

        return OnnxRvcModelDescription{m_runtimeInfo,
                                       m_modelBundle,
                                       m_sessions->generatorDescription,
                                       m_sessions->hubertDescription,
                                       m_sessions->f0Description};
    }

    [[nodiscard]] core::Expected<OnnxRvcPipelinePlan> describeConfiguredPipeline() const {
        if (m_modelBundle.modelId.empty()) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC model bundle is not configured.");
        }
        return resolvePipelinePlanFromContract(m_modelBundle.graphContract);
    }

    [[nodiscard]] core::Expected<OnnxRvcResult> convertChunk(
        const OnnxRvcRequest& request) const {
        if (m_modelBundle.modelId.empty()) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC model bundle is not configured.");
        }
        if (request.pcm16Audio.empty() || (request.pcm16Audio.size() % 2U) != 0U ||
            request.sampleRate <= 0 || request.channels <= 0) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC request audio is invalid.");
        }
        if (m_sessions == nullptr) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC ONNX sessions are not loaded.");
        }

        auto pipeline = resolvePipelinePlanFromContract(m_modelBundle.graphContract);
        if (!pipeline) {
            return pipeline.error();
        }
        auto preparedAudio = prepareNativeRvcAudio(request, m_modelBundle, pipeline.value());
        if (!preparedAudio) {
            return preparedAudio.error();
        }

        const auto startedAt = std::chrono::steady_clock::now();
        auto result = runNativePipeline(pipeline.value(), preparedAudio.value());
        if (!result) {
            return result.error();
        }
        const auto elapsed = std::chrono::steady_clock::now() - startedAt;
        result.value().latencyMs =
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                                 .count());
        return result;
    }

private:
    [[nodiscard]] core::Expected<OnnxRvcResult> runNativePipeline(
        const OnnxRvcPipelinePlan& pipeline,
        const NativeRvcPreparedAudio& preparedAudio) const {
        if (m_runtime == nullptr || m_sessions == nullptr) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Native RVC ONNX sessions are not loaded.");
        }

        const auto& api = m_runtime->api();
        auto memoryInfo = createCpuMemoryInfo(api);
        if (!memoryInfo) {
            return memoryInfo.error();
        }

        auto hubertInput = createFloatTensor(
            api,
            memoryInfo.value().get(),
            preparedAudio.hubertAudio);
        if (!hubertInput) {
            return hubertInput.error();
        }
        std::vector<std::string> hubertInputNames{preparedAudio.hubertAudio.name};
        std::vector<const OrtValue*> hubertInputs{hubertInput.value().value.get()};
        auto hubertOutput = runSingleOutputSession(
            api,
            m_sessions->hubert.get(),
            "HuBERT encoder",
            hubertInputNames,
            hubertInputs,
            pipeline.hubertContentOutput.tensor.name);
        if (!hubertOutput) {
            return hubertOutput.error();
        }
        auto contentTensor = readFloatTensor(
            api,
            hubertOutput.value().get(),
            "Read HuBERT content output");
        if (!contentTensor) {
            return contentTensor.error();
        }
        auto contentFrames = contentFrameCountFrom(contentTensor.value().dimensions);
        if (!contentFrames) {
            return contentFrames.error();
        }

        auto f0Input = createFloatTensor(api, memoryInfo.value().get(), preparedAudio.f0Audio);
        if (!f0Input) {
            return f0Input.error();
        }
        std::vector<std::string> f0InputNames{preparedAudio.f0Audio.name};
        std::vector<const OrtValue*> f0Inputs{f0Input.value().value.get()};
        auto f0Output = runSingleOutputSession(
            api,
            m_sessions->f0.get(),
            "RMVPE F0 extractor",
            f0InputNames,
            f0Inputs,
            pipeline.f0PitchOutput.tensor.name);
        if (!f0Output) {
            return f0Output.error();
        }
        auto pitchTensor = readFloatTensor(api, f0Output.value().get(), "Read F0 output");
        if (!pitchTensor) {
            return pitchTensor.error();
        }

        auto auxiliaryInputs =
            prepareGeneratorAuxiliaryInputs(pipeline, contentFrames.value());
        if (!auxiliaryInputs) {
            return auxiliaryInputs.error();
        }

        std::vector<OrtOwnedTensor> auxiliaryTensors;
        auxiliaryTensors.reserve(auxiliaryInputs.value().size());
        for (const auto& auxiliary : auxiliaryInputs.value()) {
            auto tensor = createAuxiliaryTensor(api, memoryInfo.value().get(), auxiliary);
            if (!tensor) {
                return tensor.error();
            }
            auxiliaryTensors.push_back(std::move(tensor.value()));
        }

        std::vector<std::string> generatorInputNames;
        generatorInputNames.reserve(2U + auxiliaryInputs.value().size());
        generatorInputNames.push_back(pipeline.generatorContentInput.tensor.name);
        generatorInputNames.push_back(pipeline.generatorF0Input.tensor.name);
        for (const auto& auxiliary : auxiliaryInputs.value()) {
            generatorInputNames.push_back(auxiliary.name);
        }

        std::vector<const OrtValue*> generatorInputs;
        generatorInputs.reserve(generatorInputNames.size());
        generatorInputs.push_back(hubertOutput.value().get());
        generatorInputs.push_back(f0Output.value().get());
        for (const auto& auxiliary : auxiliaryTensors) {
            generatorInputs.push_back(auxiliary.value.get());
        }

        auto generatorOutput = runSingleOutputSession(
            api,
            m_sessions->generator.get(),
            "RVC generator",
            generatorInputNames,
            generatorInputs,
            pipeline.generatorAudioOutput.tensor.name);
        if (!generatorOutput) {
            return generatorOutput.error();
        }
        auto audioTensor = readFloatTensor(
            api,
            generatorOutput.value().get(),
            "Read RVC generator output");
        if (!audioTensor) {
            return audioTensor.error();
        }

        OnnxRvcResult result;
        result.sampleRate = m_modelBundle.sampleRate;
        result.channels = 1;
        result.pcm16Audio = pcmBytesFromFloatAudio(audioTensor.value().values, result.sampleRate);
        return result;
    }

    [[nodiscard]] core::Expected<std::unique_ptr<OrtRuntimeLibrary>> loadRuntime() const {
        const auto candidates = runtimeCandidatesFor(m_runtimeDllPath);
        for (const auto& candidate : candidates) {
            if (!fileExists(candidate)) {
                continue;
            }

            HMODULE library = LoadLibraryW(candidate.wstring().c_str());
            if (library == nullptr) {
                const auto message = win32ErrorMessage(GetLastError());
                return onnxError("Failed to load " + candidate.string() + ": " + message);
            }

            const auto* symbol = GetProcAddress(library, "OrtGetApiBase");
            if (symbol == nullptr) {
                FreeLibrary(library);
                return core::makeError(core::ErrorCode::InvalidArgument,
                                       "onnxruntime.dll does not export OrtGetApiBase.");
            }

#pragma warning(push)
#pragma warning(disable : 4191)
            const auto getApiBase = reinterpret_cast<OrtGetApiBaseFn>(symbol);
#pragma warning(pop)
            const OrtApiBase* apiBase = getApiBase();
            if (apiBase == nullptr || apiBase->GetApi == nullptr) {
                FreeLibrary(library);
                return core::makeError(core::ErrorCode::InvalidArgument,
                                       "onnxruntime.dll returned an invalid API base.");
            }

            const OrtApi* api = apiBase->GetApi(ORT_API_VERSION);
            if (api == nullptr) {
                FreeLibrary(library);
                return core::makeError(
                    core::ErrorCode::InvalidArgument,
                    "onnxruntime.dll does not support ORT API version " +
                        std::to_string(ORT_API_VERSION) + ".");
            }

            const char* version = nullptr;
            if (apiBase->GetVersionString != nullptr) {
                version = apiBase->GetVersionString();
            }

            auto runtime = std::make_unique<OrtRuntimeLibrary>(
                library,
                api,
                candidate,
                version == nullptr ? std::string{} : std::string{version});
            return core::Expected<std::unique_ptr<OrtRuntimeLibrary>>{std::move(runtime)};
        }

        return onnxError("onnxruntime.dll was not found beside the app or under "
                         "%LOCALAPPDATA%/VoxStudio/onnxruntime/.");
    }

    [[nodiscard]] core::Expected<OrtSessionPtr> createSession(
        const OrtRuntimeLibrary& runtime,
        const OrtEnv* env,
        const OrtSessionOptions* options,
        const std::filesystem::path& modelPath,
        const std::string& label) const {
        OrtSession* session = nullptr;
        const auto& api = runtime.api();
        auto created = ortStatusToExpected(
            api,
            api.CreateSession(env, modelPath.wstring().c_str(), options, &session),
            "Create " + label + " ONNX session");
        if (!created) {
            return created.error();
        }

        OrtSessionPtr sessionPtr{session, OrtSessionDeleter{&api}};
        return core::Expected<OrtSessionPtr>{std::move(sessionPtr)};
    }

    [[nodiscard]] core::Expected<std::string> readIoName(const OrtApi& api,
                                                         OrtAllocator* allocator,
                                                         const OrtSession* session,
                                                         const size_t index,
                                                         const bool input) const {
        char* rawName = nullptr;
        auto nameLoaded = ortStatusToExpected(
            api,
            input ? api.SessionGetInputName(session, index, allocator, &rawName)
                  : api.SessionGetOutputName(session, index, allocator, &rawName),
            input ? "Read ONNX input name" : "Read ONNX output name");
        if (!nameLoaded) {
            return nameLoaded.error();
        }

        std::string name = rawName == nullptr ? std::string{} : std::string{rawName};
        if (rawName != nullptr) {
            auto released = ortStatusToExpected(
                api,
                api.AllocatorFree(allocator, rawName),
                "Release ONNX name allocation");
            if (!released) {
                return released.error();
            }
        }
        return name;
    }

    [[nodiscard]] core::Expected<OnnxTensorDescription> describeTensor(
        const OrtApi& api,
        OrtAllocator* allocator,
        const OrtSession* session,
        const size_t index,
        const bool input) const {
        auto name = readIoName(api, allocator, session, index, input);
        if (!name) {
            return name.error();
        }

        OrtTypeInfo* rawTypeInfo = nullptr;
        auto typeInfoLoaded = ortStatusToExpected(
            api,
            input ? api.SessionGetInputTypeInfo(session, index, &rawTypeInfo)
                  : api.SessionGetOutputTypeInfo(session, index, &rawTypeInfo),
            input ? "Read ONNX input type info" : "Read ONNX output type info");
        if (!typeInfoLoaded) {
            return typeInfoLoaded.error();
        }
        OrtTypeInfoPtr typeInfo{rawTypeInfo, OrtTypeInfoDeleter{&api}};

        ONNXType onnxType = ONNX_TYPE_UNKNOWN;
        auto onnxTypeRead = ortStatusToExpected(
            api,
            api.GetOnnxTypeFromTypeInfo(typeInfo.get(), &onnxType),
            "Read ONNX value type");
        if (!onnxTypeRead) {
            return onnxTypeRead.error();
        }

        OnnxTensorDescription description;
        description.name = std::move(name.value());
        description.elementType = onnxTypeName(onnxType);
        if (onnxType != ONNX_TYPE_TENSOR) {
            return description;
        }

        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        auto tensorInfoRead = ortStatusToExpected(
            api,
            api.CastTypeInfoToTensorInfo(typeInfo.get(), &tensorInfo),
            "Read ONNX tensor type info");
        if (!tensorInfoRead) {
            return tensorInfoRead.error();
        }
        if (tensorInfo == nullptr) {
            return description;
        }

        ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        auto elementTypeRead = ortStatusToExpected(
            api,
            api.GetTensorElementType(tensorInfo, &elementType),
            "Read ONNX tensor element type");
        if (!elementTypeRead) {
            return elementTypeRead.error();
        }
        description.elementType = tensorElementTypeName(elementType);

        size_t dimensionCount = 0;
        auto dimensionCountRead = ortStatusToExpected(
            api,
            api.GetDimensionsCount(tensorInfo, &dimensionCount),
            "Read ONNX tensor dimension count");
        if (!dimensionCountRead) {
            return dimensionCountRead.error();
        }

        description.dimensions.resize(dimensionCount);
        auto dimensionsRead = ortStatusToExpected(
            api,
            api.GetDimensions(tensorInfo, description.dimensions.data(), dimensionCount),
            "Read ONNX tensor dimensions");
        if (!dimensionsRead) {
            return dimensionsRead.error();
        }

        std::vector<const char*> symbolicDimensions(dimensionCount, nullptr);
        auto symbolicRead = ortStatusToExpected(
            api,
            api.GetSymbolicDimensions(
                tensorInfo,
                symbolicDimensions.data(),
                symbolicDimensions.size()),
            "Read ONNX tensor symbolic dimensions");
        if (!symbolicRead) {
            return symbolicRead.error();
        }
        description.symbolicDimensions.reserve(symbolicDimensions.size());
        for (const char* symbolic : symbolicDimensions) {
            description.symbolicDimensions.emplace_back(symbolic == nullptr ? "" : symbolic);
        }

        return description;
    }

    [[nodiscard]] core::Expected<std::vector<OnnxTensorDescription>> describeTensors(
        const OrtApi& api,
        OrtAllocator* allocator,
        const OrtSession* session,
        const bool inputs) const {
        size_t count = 0;
        auto countRead = ortStatusToExpected(
            api,
            inputs ? api.SessionGetInputCount(session, &count)
                   : api.SessionGetOutputCount(session, &count),
            inputs ? "Read ONNX input count" : "Read ONNX output count");
        if (!countRead) {
            return countRead.error();
        }

        std::vector<OnnxTensorDescription> tensors;
        tensors.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            auto tensor = describeTensor(api, allocator, session, index, inputs);
            if (!tensor) {
                return tensor.error();
            }
            tensors.push_back(std::move(tensor.value()));
        }
        return tensors;
    }

    [[nodiscard]] core::Expected<OnnxSessionDescription> describeSession(
        const OrtRuntimeLibrary& runtime,
        OrtAllocator* allocator,
        const OrtSession* session,
        std::string label) const {
        const auto& api = runtime.api();
        auto inputs = describeTensors(api, allocator, session, true);
        if (!inputs) {
            return inputs.error();
        }
        auto outputs = describeTensors(api, allocator, session, false);
        if (!outputs) {
            return outputs.error();
        }

        OnnxSessionDescription description;
        description.label = std::move(label);
        description.inputs = std::move(inputs.value());
        description.outputs = std::move(outputs.value());
        return description;
    }

    [[nodiscard]] core::Expected<std::unique_ptr<NativeSessions>> createSessions(
        const OrtRuntimeLibrary& runtime,
        const OnnxRvcModelBundle& bundle) const {
        const auto& api = runtime.api();

        OrtEnv* env = nullptr;
        auto envCreated = ortStatusToExpected(
            api,
            api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "VoxStudioNativeRvc", &env),
            "Create ONNX Runtime environment");
        if (!envCreated) {
            return envCreated.error();
        }
        OrtEnvPtr envPtr{env, OrtEnvDeleter{&api}};

        auto telemetryDisabled = ortStatusToExpected(
            api,
            api.DisableTelemetryEvents(envPtr.get()),
            "Disable ORT telemetry");
        if (!telemetryDisabled) {
            return telemetryDisabled.error();
        }

        OrtSessionOptions* options = nullptr;
        auto optionsCreated = ortStatusToExpected(
            api,
            api.CreateSessionOptions(&options),
            "Create ONNX Runtime session options");
        if (!optionsCreated) {
            return optionsCreated.error();
        }
        OrtSessionOptionsPtr optionsPtr{options, OrtSessionOptionsDeleter{&api}};

        auto optimized = ortStatusToExpected(
            api,
            api.SetSessionGraphOptimizationLevel(optionsPtr.get(), ORT_ENABLE_ALL),
            "Enable ONNX graph optimizations");
        if (!optimized) {
            return optimized.error();
        }

        auto generator = createSession(
            runtime,
            envPtr.get(),
            optionsPtr.get(),
            bundle.generatorModelPath,
            "RVC generator");
        if (!generator) {
            return generator.error();
        }
        auto hubert = createSession(
            runtime,
            envPtr.get(),
            optionsPtr.get(),
            bundle.hubertModelPath,
            "HuBERT encoder");
        if (!hubert) {
            return hubert.error();
        }
        auto f0 = createSession(
            runtime,
            envPtr.get(),
            optionsPtr.get(),
            bundle.f0ModelPath,
            "RMVPE F0 extractor");
        if (!f0) {
            return f0.error();
        }

        OrtAllocator* allocator = nullptr;
        auto allocatorLoaded = ortStatusToExpected(
            api,
            api.GetAllocatorWithDefaultOptions(&allocator),
            "Read ONNX Runtime default allocator");
        if (!allocatorLoaded) {
            return allocatorLoaded.error();
        }

        auto generatorDescription = describeSession(
            runtime,
            allocator,
            generator.value().get(),
            "RVC generator");
        if (!generatorDescription) {
            return generatorDescription.error();
        }
        auto hubertDescription = describeSession(
            runtime,
            allocator,
            hubert.value().get(),
            "HuBERT encoder");
        if (!hubertDescription) {
            return hubertDescription.error();
        }
        auto f0Description = describeSession(
            runtime,
            allocator,
            f0.value().get(),
            "RMVPE F0 extractor");
        if (!f0Description) {
            return f0Description.error();
        }

        const OnnxRvcModelDescription description{runtime.info(),
                                                  bundle,
                                                  generatorDescription.value(),
                                                  hubertDescription.value(),
                                                  f0Description.value()};
        auto contractValid =
            validateGraphContractAgainstDescription(bundle.graphContract, description);
        if (!contractValid) {
            return contractValid.error();
        }

        auto sessions = std::make_unique<NativeSessions>();
        sessions->env = std::move(envPtr);
        sessions->options = std::move(optionsPtr);
        sessions->generator = std::move(generator.value());
        sessions->hubert = std::move(hubert.value());
        sessions->f0 = std::move(f0.value());
        sessions->generatorDescription = std::move(generatorDescription.value());
        sessions->hubertDescription = std::move(hubertDescription.value());
        sessions->f0Description = std::move(f0Description.value());
        return core::Expected<std::unique_ptr<NativeSessions>>{std::move(sessions)};
    }

    std::filesystem::path m_runtimeDllPath;
    OnnxRuntimeInfo m_runtimeInfo;
    OnnxRvcModelBundle m_modelBundle;
    std::unique_ptr<OrtRuntimeLibrary> m_runtime;
    std::unique_ptr<NativeSessions> m_sessions;
};

OnnxRvcEngine::OnnxRvcEngine()
    : OnnxRvcEngine(std::filesystem::path{}) {}

OnnxRvcEngine::OnnxRvcEngine(std::filesystem::path runtimeDllPath)
    : m_impl(std::make_unique<Impl>(std::move(runtimeDllPath))) {}

OnnxRvcEngine::~OnnxRvcEngine() = default;
OnnxRvcEngine::OnnxRvcEngine(OnnxRvcEngine&&) noexcept = default;
OnnxRvcEngine& OnnxRvcEngine::operator=(OnnxRvcEngine&&) noexcept = default;

std::vector<std::filesystem::path> OnnxRvcEngine::defaultRuntimeDllCandidates() {
    return runtimeCandidatesFor({});
}

std::filesystem::path OnnxRvcEngine::defaultNativeModelRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "rvc_onnx_models";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "rvc_onnx_models";
}

core::Expected<bool> OnnxRvcEngine::validateGraphContract(
    const OnnxRvcGraphContract& contract,
    const OnnxRvcModelDescription& description) {
    return validateGraphContractAgainstDescription(contract, description);
}

core::Expected<OnnxRvcPipelinePlan> OnnxRvcEngine::resolvePipelinePlan(
    const OnnxRvcGraphContract& contract) {
    return resolvePipelinePlanFromContract(contract);
}

core::Expected<OnnxRuntimeInfo> OnnxRvcEngine::probeRuntime() const {
    return m_impl->probeRuntime();
}

core::Expected<OnnxRvcModelBundle> OnnxRvcEngine::loadModelBundle(
    const std::filesystem::path& bundleRoot) const {
    return m_impl->loadModelBundle(bundleRoot);
}

core::Expected<bool> OnnxRvcEngine::configureModelBundle(OnnxRvcModelBundle bundle) {
    return m_impl->configureModelBundle(std::move(bundle));
}

core::Expected<OnnxRvcModelDescription> OnnxRvcEngine::describeConfiguredModel() const {
    return m_impl->describeConfiguredModel();
}

core::Expected<OnnxRvcPipelinePlan> OnnxRvcEngine::describeConfiguredPipeline() const {
    return m_impl->describeConfiguredPipeline();
}

core::Expected<OnnxRvcResult> OnnxRvcEngine::convertChunk(
    const OnnxRvcRequest& request) const {
    return m_impl->convertChunk(request);
}

} // namespace voxstudio::rvc
