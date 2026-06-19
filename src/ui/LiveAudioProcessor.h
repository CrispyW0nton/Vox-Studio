#pragma once

#include "audio/Capture.h"
#include "dsp/NoiseSuppressor.h"
#include "dsp/Vad.h"

#include <QByteArray>
#include <QObject>

#include <memory>

class QTimer;

namespace voxstudio::ui {

class LiveAudioProcessor final : public QObject {
    Q_OBJECT

public:
    explicit LiveAudioProcessor(QObject* parent = nullptr);
    ~LiveAudioProcessor() override;

    void start(audio::Capture* capture);
    void stop();
    void setPassthroughEnabled(bool enabled);
    void setCloudCaptureEnabled(bool enabled);
    void setLocalRvcCaptureEnabled(bool enabled);
    void flushCloudCapture();
    void flushLocalRvcCapture();

signals:
    void meterUpdated(int level, bool speechActive);
    void statusMessage(QString message);
    void cloudPcmChunkReady(QByteArray chunk);
    void localRvcPcmChunkReady(QByteArray chunk);

private:
    void processOnce();
    void appendCloudFrame(const audio::AudioFrame& frame);
    void appendLocalRvcFrame(const audio::AudioFrame& frame);
    void emitCloudChunk();
    void emitLocalRvcChunk();

    audio::Capture* m_capture{nullptr};
    dsp::NoiseSuppressor m_noiseSuppressor;
    dsp::Vad m_vad;
    std::unique_ptr<QTimer> m_timer;
    QByteArray m_cloudPcmBuffer;
    QByteArray m_localRvcPcmBuffer;
    bool m_passthroughEnabled{true};
    bool m_cloudCaptureEnabled{false};
    bool m_localRvcCaptureEnabled{false};
};

} // namespace voxstudio::ui
