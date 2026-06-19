#include "ui/LiveAudioProcessor.h"

#include <QTimer>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace voxstudio::ui {
namespace {

constexpr int kProcessorIntervalMs = 10;
constexpr int kMaxFramesPerTick = 16;
constexpr int kCloudSampleRate = 16000;
constexpr int kCloudSourceSampleRate = audio::kRealtimeSampleRate;
constexpr int kCloudDownsampleRatio = kCloudSourceSampleRate / kCloudSampleRate;
constexpr qsizetype kCloudChunkBytes = kCloudSampleRate * 2;
constexpr int kLocalRvcFrameMs = 20;
constexpr qsizetype kLocalRvcChunkBytes =
    (audio::kRealtimeSampleRate * kLocalRvcFrameMs / 1000) * 2;

[[nodiscard]] std::int16_t floatToPcm16(const float sample) noexcept {
    const auto clamped = std::clamp(sample, -1.0F, 1.0F);
    const auto scaled = static_cast<int>(std::lround(clamped * 32767.0F));
    return static_cast<std::int16_t>(
        std::clamp(scaled,
                   static_cast<int>(std::numeric_limits<std::int16_t>::min()),
                   static_cast<int>(std::numeric_limits<std::int16_t>::max())));
}

} // namespace

LiveAudioProcessor::LiveAudioProcessor(QObject* parent)
    : QObject(parent) {}

LiveAudioProcessor::~LiveAudioProcessor() {
    stop();
}

void LiveAudioProcessor::start(audio::Capture* capture) {
    m_capture = capture;
    m_noiseSuppressor = dsp::NoiseSuppressor{};
    m_vad = dsp::Vad{};
    m_cloudPcmBuffer.clear();
    m_localRvcPcmBuffer.clear();

    if (!m_timer) {
        m_timer = std::make_unique<QTimer>();
        m_timer->setInterval(kProcessorIntervalMs);
        connect(m_timer.get(), &QTimer::timeout, this, &LiveAudioProcessor::processOnce);
    }

    m_timer->start();
}

void LiveAudioProcessor::stop() {
    if (m_timer) {
        m_timer->stop();
    }
    flushCloudCapture();
    flushLocalRvcCapture();
    m_capture = nullptr;
    emit meterUpdated(0, false);
}

void LiveAudioProcessor::setPassthroughEnabled(const bool enabled) {
    m_passthroughEnabled = enabled;
}

void LiveAudioProcessor::setCloudCaptureEnabled(const bool enabled) {
    if (!enabled) {
        flushCloudCapture();
    } else {
        m_cloudPcmBuffer.clear();
        m_cloudPcmBuffer.reserve(kCloudChunkBytes);
    }
    m_cloudCaptureEnabled = enabled;
}

void LiveAudioProcessor::setLocalRvcCaptureEnabled(const bool enabled) {
    if (!enabled) {
        flushLocalRvcCapture();
    } else {
        m_localRvcPcmBuffer.clear();
        m_localRvcPcmBuffer.reserve(kLocalRvcChunkBytes);
    }
    m_localRvcCaptureEnabled = enabled;
}

void LiveAudioProcessor::flushCloudCapture() {
    if (!m_cloudPcmBuffer.isEmpty()) {
        emitCloudChunk();
    }
}

void LiveAudioProcessor::flushLocalRvcCapture() {
    if (!m_localRvcPcmBuffer.isEmpty()) {
        emitLocalRvcChunk();
    }
}

void LiveAudioProcessor::processOnce() {
    if (m_capture == nullptr) {
        return;
    }

    audio::AudioFrame frame;
    float peakRms = 0.0F;
    bool speechActive = false;
    int processed = 0;

    while (processed < kMaxFramesPerTick && m_capture->tryPopCapturedFrame(frame)) {
        auto cleaned = m_noiseSuppressor.processFrame(frame);
        if (!cleaned) {
            emit statusMessage(QString::fromStdString(cleaned.error().message));
            ++processed;
            continue;
        }

        const auto vad = m_vad.analyze(cleaned.value());
        speechActive = speechActive || vad.speechActive;
        peakRms = std::max(peakRms, cleaned.value().rms);
        if (m_passthroughEnabled && !m_capture->tryPushMonitorFrame(cleaned.value())) {
            emit statusMessage(QStringLiteral("Monitor queue is full; dropping a frame."));
        }
        if (m_cloudCaptureEnabled) {
            appendCloudFrame(cleaned.value());
        }
        if (m_localRvcCaptureEnabled) {
            appendLocalRvcFrame(cleaned.value());
        }
        ++processed;
    }

    const auto stats = m_capture->stats();
    peakRms = std::max(peakRms, stats.inputRms);
    emit meterUpdated(std::clamp(static_cast<int>(peakRms * 300.0F), 0, 100), speechActive);
}

void LiveAudioProcessor::appendCloudFrame(const audio::AudioFrame& frame) {
    if (frame.sampleRate != kCloudSourceSampleRate || frame.channels != audio::kRealtimeChannels) {
        emit statusMessage(QStringLiteral("Cloud capture requires 48 kHz mono input."));
        return;
    }

    std::array<char, 2> bytes{};
    for (std::size_t index = 0; index + 2U < frame.frameCount;
         index += static_cast<std::size_t>(kCloudDownsampleRatio)) {
        const auto averaged =
            (frame.samples[index] + frame.samples[index + 1U] + frame.samples[index + 2U]) /
            static_cast<float>(kCloudDownsampleRatio);
        const auto sample = floatToPcm16(averaged);
        bytes[0] = static_cast<char>(sample & 0xFF);
        bytes[1] = static_cast<char>((sample >> 8) & 0xFF);
        m_cloudPcmBuffer.append(bytes.data(), static_cast<qsizetype>(bytes.size()));
    }

    while (m_cloudPcmBuffer.size() >= kCloudChunkBytes) {
        emit cloudPcmChunkReady(m_cloudPcmBuffer.left(kCloudChunkBytes));
        m_cloudPcmBuffer.remove(0, kCloudChunkBytes);
    }
}

void LiveAudioProcessor::emitCloudChunk() {
    emit cloudPcmChunkReady(m_cloudPcmBuffer);
    m_cloudPcmBuffer.clear();
}

void LiveAudioProcessor::appendLocalRvcFrame(const audio::AudioFrame& frame) {
    if (frame.sampleRate != audio::kRealtimeSampleRate ||
        frame.channels != audio::kRealtimeChannels) {
        emit statusMessage(QStringLiteral("Local RVC requires 48 kHz mono input."));
        return;
    }

    std::array<char, 2> bytes{};
    for (std::size_t index = 0; index < frame.frameCount; ++index) {
        const auto sample = floatToPcm16(frame.samples[index]);
        bytes[0] = static_cast<char>(sample & 0xFF);
        bytes[1] = static_cast<char>((sample >> 8) & 0xFF);
        m_localRvcPcmBuffer.append(bytes.data(), static_cast<qsizetype>(bytes.size()));
    }

    while (m_localRvcPcmBuffer.size() >= kLocalRvcChunkBytes) {
        emit localRvcPcmChunkReady(m_localRvcPcmBuffer.left(kLocalRvcChunkBytes));
        m_localRvcPcmBuffer.remove(0, kLocalRvcChunkBytes);
    }
}

void LiveAudioProcessor::emitLocalRvcChunk() {
    emit localRvcPcmChunkReady(m_localRvcPcmBuffer);
    m_localRvcPcmBuffer.clear();
}

} // namespace voxstudio::ui
