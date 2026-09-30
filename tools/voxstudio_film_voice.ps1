param(
    [ValidateSet('Preflight', 'GetVoice', 'DownloadVoiceSample', 'DeleteVoice', 'SearchSharedVoices', 'AddSharedVoice', 'DesignPreview', 'CreateVoice', 'Synthesize', 'SynthesizeWithTimestamps', 'Dialogue', 'SpeechToSpeech', 'SoundEffect', 'Music')]
    [string]$Mode = 'Preflight',
    [string]$VoiceDescription,
    [string]$PreviewText,
    [string]$GeneratedVoiceId,
    [string]$PublicOwnerId,
    [string]$VoiceName,
    [string]$VoiceId,
    [string]$SampleId,
    [string]$Text,
    [string]$InputAudioPath,
    [string]$DialogueInputPath,
    [string]$AlignmentOutputPath,
    [string]$LibrarySearch,
    [string]$LibraryGender,
    [string]$LibraryAge,
    [string]$LibraryAccent,
    [string]$LibraryLanguage = 'en',
    [string]$LibraryUseCase,
    [ValidateRange(1, 100)]
    [int]$LibraryPageSize = 30,
    [string]$OutputDirectory,
    [string]$OutputPath,
    [string]$ModelId = 'eleven_v3',
    [string]$StsModelId = 'eleven_multilingual_sts_v2',
    [ValidateSet('pcm_48000', 'mp3_44100_128')]
    [string]$StsOutputFormat = 'pcm_48000',
    [ValidateRange(0.0, 1.0)]
    [double]$Stability = 0.35,
    [ValidateRange(0.0, 1.0)]
    [double]$SimilarityBoost = 0.75,
    [ValidateRange(0.0, 1.0)]
    [double]$Style = 0.20,
    [ValidateRange(0.7, 1.2)]
    [double]$Speed = 1.0,
    [ValidateRange(0.5, 30.0)]
    [double]$DurationSeconds = 5.0,
    [ValidateRange(0.0, 1.0)]
    [double]$PromptInfluence = 0.3,
    [ValidateRange(3000, 600000)]
    [int]$MusicLengthMs = 30000,
    [switch]$LoopSound,
    [switch]$UseSpeakerBoost,
    [switch]$RemoveBackgroundNoise,
    [int]$Seed = 0,
    [switch]$ConfirmCreditSpend,
    [string]$ConfirmDeleteVoiceId
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$apiBase = 'https://api.elevenlabs.io/v1'
$secretPath = Join-Path $env:LOCALAPPDATA 'VoxStudio\secrets\elevenlabs.bin'
$entropyText = 'VoxStudio.ElevenLabs.ApiKey.v1'
$validatedPrefix = "VoxStudio.ElevenLabs.Validated.v1`n"

function Get-VoxStudioApiKey {
    if (-not (Test-Path -LiteralPath $secretPath -PathType Leaf)) {
        throw 'Vox Studio does not have a saved ElevenLabs credential.'
    }

    Add-Type -AssemblyName System.Security.Cryptography.ProtectedData
    $encrypted = [System.IO.File]::ReadAllBytes($secretPath)
    $entropy = [System.Text.Encoding]::UTF8.GetBytes($entropyText)
    $decrypted = [System.Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted,
        $entropy,
        [System.Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    try {
        $payload = [System.Text.Encoding]::UTF8.GetString($decrypted)
        if (-not $payload.StartsWith($validatedPrefix, [System.StringComparison]::Ordinal)) {
            throw 'The saved Vox Studio credential is not a validated payload.'
        }
        $key = $payload.Substring($validatedPrefix.Length)
        if ([string]::IsNullOrWhiteSpace($key)) {
            throw 'The saved Vox Studio credential is empty.'
        }
        return $key
    }
    finally {
        [Array]::Clear($decrypted, 0, $decrypted.Length)
    }
}

function Invoke-ElevenLabsJson {
    param(
        [Parameter(Mandatory)] [string]$Method,
        [Parameter(Mandatory)] [string]$Path,
        [object]$Body
    )

    $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'application/json' }
    $parameters = @{
        Method = $Method
        Uri = "$apiBase$Path"
        Headers = $headers
        TimeoutSec = 60
    }
    if ($null -ne $Body) {
        $parameters.ContentType = 'application/json'
        $parameters.Body = $Body | ConvertTo-Json -Depth 12 -Compress
    }
    return Invoke-RestMethod @parameters
}

function Assert-CreditSpendConfirmed {
    if (-not $ConfirmCreditSpend) {
        throw "$Mode spends ElevenLabs credits. Re-run with -ConfirmCreditSpend after reviewing Preflight output and the exact text length."
    }
}

function Write-JsonResult {
    param([Parameter(Mandatory)] [object]$Value)
    $Value | ConvertTo-Json -Depth 12
}

$script:apiKey = Get-VoxStudioApiKey
try {
    switch ($Mode) {
        'Preflight' {
            $subscription = Invoke-ElevenLabsJson -Method GET -Path '/user/subscription'
            $voicesResponse = Invoke-ElevenLabsJson -Method GET -Path '/voices'
            $modelsResponse = Invoke-ElevenLabsJson -Method GET -Path '/models'

            $voices = @($voicesResponse.voices | ForEach-Object {
                [ordered]@{
                    voice_id = $_.voice_id
                    name = $_.name
                    category = $_.category
                    description = $_.description
                    labels = $_.labels
                    is_owner = $_.is_owner
                }
            })
            $models = @($modelsResponse | Where-Object { $_.can_do_text_to_speech } | ForEach-Object {
                [ordered]@{
                    model_id = $_.model_id
                    name = $_.name
                    can_use_style = $_.can_use_style
                    can_use_speaker_boost = $_.can_use_speaker_boost
                }
            })
            $voiceConversionModels = @($modelsResponse | Where-Object { $_.can_do_voice_conversion } | ForEach-Object {
                [ordered]@{
                    model_id = $_.model_id
                    name = $_.name
                    can_use_speaker_boost = $_.can_use_speaker_boost
                    description = $_.description
                }
            })
            $remaining = [Math]::Max(0, [int64]$subscription.character_limit - [int64]$subscription.character_count)

            Write-JsonResult ([ordered]@{
                credential_source = 'Vox Studio DPAPI vault'
                credential_exposed = $false
                tier = $subscription.tier
                subscription_status = $subscription.status
                credits_used = $subscription.character_count
                credits_limit = $subscription.character_limit
                credits_remaining = $remaining
                voice_slots_used = $subscription.voice_slots_used
                voice_limit = $subscription.voice_limit
                voices = $voices
                text_to_speech_models = $models
                voice_conversion_models = $voiceConversionModels
            })
        }

        'GetVoice' {
            if ([string]::IsNullOrWhiteSpace($VoiceId)) {
                throw 'VoiceId is required for GetVoice.'
            }
            $voice = Invoke-ElevenLabsJson -Method GET -Path ("/voices/{0}" -f $VoiceId)
            Write-JsonResult ([ordered]@{
                action = 'voice_inspected'
                read_only = $true
                voice_id = $voice.voice_id
                name = $voice.name
                category = $voice.category
                description = $voice.description
                labels = $voice.labels
                is_owner = $voice.is_owner
                samples = @($voice.samples | ForEach-Object {
                    [ordered]@{
                        sample_id = $_.sample_id
                        file_name = $_.file_name
                        mime_type = $_.mime_type
                        size_bytes = $_.size_bytes
                        hash = $_.hash
                    }
                })
            })
        }

        'DownloadVoiceSample' {
            if ([string]::IsNullOrWhiteSpace($VoiceId) -or [string]::IsNullOrWhiteSpace($SampleId) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'VoiceId, SampleId, and OutputPath are required for DownloadVoiceSample.'
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'audio/mpeg' }
            Invoke-WebRequest -Method GET -Uri ("$apiBase/voices/{0}/samples/{1}/audio" -f $VoiceId,$SampleId) -Headers $headers -OutFile $OutputPath -TimeoutSec 120
            Write-JsonResult ([ordered]@{
                action = 'voice_sample_downloaded'
                read_only = $true
                voice_id = $VoiceId
                sample_id = $SampleId
                output_path = $OutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }

        'DeleteVoice' {
            if ([string]::IsNullOrWhiteSpace($VoiceId)) {
                throw 'VoiceId is required for DeleteVoice.'
            }
            if ([string]::IsNullOrWhiteSpace($ConfirmDeleteVoiceId) -or $ConfirmDeleteVoiceId -cne $VoiceId) {
                throw 'DeleteVoice requires -ConfirmDeleteVoiceId to exactly match VoiceId.'
            }
            $voice = Invoke-ElevenLabsJson -Method GET -Path ("/voices/{0}" -f $VoiceId)
            $voicesResponse = Invoke-ElevenLabsJson -Method GET -Path '/voices'
            $accountVoice = @($voicesResponse.voices | Where-Object voice_id -eq $VoiceId | Select-Object -First 1)
            if ($accountVoice.Count -ne 1 -or -not [bool]$accountVoice[0].is_owner) {
                throw 'DeleteVoice refuses to delete a voice that is not owned by the authenticated account.'
            }
            $response = Invoke-ElevenLabsJson -Method DELETE -Path ("/voices/{0}" -f $VoiceId)
            Write-JsonResult ([ordered]@{
                action = 'owned_voice_deleted'
                voice_id = $VoiceId
                name = $voice.name
                category = $voice.category
                status = $response.status
            })
        }

        'SearchSharedVoices' {
            # Read-only search of ElevenLabs' professional Voice Library. This does
            # not add a voice to the workspace, consume a voice slot, or spend
            # synthesis credits.
            $query = [System.Collections.Generic.List[string]]::new()
            $query.Add("page_size=$LibraryPageSize")
            $query.Add('category=professional')
            $query.Add('sort=usage_character_count_1y')
            if (-not [string]::IsNullOrWhiteSpace($LibrarySearch)) {
                $query.Add('search=' + [Uri]::EscapeDataString($LibrarySearch))
            }
            if (-not [string]::IsNullOrWhiteSpace($LibraryGender)) {
                $query.Add('gender=' + [Uri]::EscapeDataString($LibraryGender))
            }
            if (-not [string]::IsNullOrWhiteSpace($LibraryAge)) {
                $query.Add('age=' + [Uri]::EscapeDataString($LibraryAge))
            }
            if (-not [string]::IsNullOrWhiteSpace($LibraryAccent)) {
                $query.Add('accent=' + [Uri]::EscapeDataString($LibraryAccent))
            }
            if (-not [string]::IsNullOrWhiteSpace($LibraryLanguage)) {
                $query.Add('language=' + [Uri]::EscapeDataString($LibraryLanguage))
            }
            if (-not [string]::IsNullOrWhiteSpace($LibraryUseCase)) {
                $query.Add('use_cases=' + [Uri]::EscapeDataString($LibraryUseCase))
            }

            $response = Invoke-ElevenLabsJson -Method GET -Path ('/shared-voices?' + ($query -join '&'))
            $voices = @($response.voices | ForEach-Object {
                [ordered]@{
                    public_owner_id = $_.public_owner_id
                    voice_id = $_.voice_id
                    name = $_.name
                    accent = $_.accent
                    gender = $_.gender
                    age = $_.age
                    descriptive = $_.descriptive
                    use_case = $_.use_case
                    category = $_.category
                    language = $_.language
                    description = $_.description
                    preview_url = $_.preview_url
                    rate = $_.rate
                    notice_period = $_.notice_period
                    live_moderation_enabled = $_.live_moderation_enabled
                }
            })
            Write-JsonResult ([ordered]@{
                action = 'shared_voice_search'
                read_only = $true
                total_count = $response.total_count
                has_more = $response.has_more
                voices = $voices
            })
        }

        'AddSharedVoice' {
            if ([string]::IsNullOrWhiteSpace($PublicOwnerId) -or [string]::IsNullOrWhiteSpace($VoiceId) -or [string]::IsNullOrWhiteSpace($VoiceName)) {
                throw 'PublicOwnerId, VoiceId, and VoiceName are required for AddSharedVoice.'
            }
            $voice = Invoke-ElevenLabsJson -Method POST -Path ("/voices/add/{0}/{1}" -f $PublicOwnerId,$VoiceId) -Body ([ordered]@{
                new_name = $VoiceName
                bookmarked = $true
            })
            Write-JsonResult ([ordered]@{
                action = 'shared_voice_added'
                voice_id = $voice.voice_id
                name = $VoiceName
                source = 'ElevenLabs professional Voice Library'
            })
        }

        'DesignPreview' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($VoiceDescription) -or $VoiceDescription.Length -lt 20) {
                throw 'VoiceDescription must contain at least 20 characters.'
            }
            if ([string]::IsNullOrWhiteSpace($PreviewText) -or $PreviewText.Length -lt 100 -or $PreviewText.Length -gt 1000) {
                throw 'PreviewText must contain 100-1000 characters.'
            }
            if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
                throw 'OutputDirectory is required for DesignPreview.'
            }
            New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

            $body = [ordered]@{
                voice_description = $VoiceDescription
                text = $PreviewText
                auto_generate_text = $false
                quality = 0.90
                guidance_scale = 4.0
                loudness = 0.0
                seed = $Seed
            }
            $response = Invoke-ElevenLabsJson -Method POST -Path '/text-to-voice/create-previews?output_format=mp3_44100_192' -Body $body
            $previewRecords = @()
            $index = 0
            foreach ($preview in @($response.previews)) {
                $index++
                $previewPath = Join-Path $OutputDirectory ("preview_{0:D2}.mp3" -f $index)
                [System.IO.File]::WriteAllBytes($previewPath, [Convert]::FromBase64String($preview.audio_base_64))
                $previewRecords += [ordered]@{
                    index = $index
                    generated_voice_id = $preview.generated_voice_id
                    media_type = $preview.media_type
                    duration_seconds = $preview.duration_secs
                    path = $previewPath
                    sha256 = (Get-FileHash -LiteralPath $previewPath -Algorithm SHA256).Hash
                }
            }
            Write-JsonResult ([ordered]@{
                action = 'voice_design_preview'
                charged_characters = $PreviewText.Length
                prompt = $VoiceDescription
                preview_text = $response.text
                previews = $previewRecords
            })
        }

        'CreateVoice' {
            if ([string]::IsNullOrWhiteSpace($GeneratedVoiceId) -or [string]::IsNullOrWhiteSpace($VoiceName) -or [string]::IsNullOrWhiteSpace($VoiceDescription)) {
                throw 'GeneratedVoiceId, VoiceName, and VoiceDescription are required for CreateVoice.'
            }
            $body = [ordered]@{
                voice_name = $VoiceName
                voice_description = $VoiceDescription
                generated_voice_id = $GeneratedVoiceId
                labels = @{ project = 'Epic of Gilgamesh'; source = 'Vox Studio Voice Design' }
            }
            $voice = Invoke-ElevenLabsJson -Method POST -Path '/text-to-voice' -Body $body
            Write-JsonResult ([ordered]@{
                action = 'voice_created'
                voice_id = $voice.voice_id
                name = $voice.name
                category = $voice.category
            })
        }

        'Synthesize' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($VoiceId) -or [string]::IsNullOrWhiteSpace($Text) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'VoiceId, Text, and OutputPath are required for Synthesize.'
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            if ($StsOutputFormat -eq 'pcm_48000' -and [System.IO.Path]::GetExtension($OutputPath) -ne '.wav') {
                throw 'SpeechToSpeech with pcm_48000 requires OutputPath to use the .wav extension.'
            }
            $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'application/octet-stream' }
            $voiceSettings = if ($ModelId -like 'eleven_v3*') {
                [ordered]@{ stability = $Stability }
            } else {
                [ordered]@{
                    stability = $Stability
                    similarity_boost = $SimilarityBoost
                    style = $Style
                    use_speaker_boost = [bool]$UseSpeakerBoost
                    speed = $Speed
                }
            }
            $body = [ordered]@{
                text = $Text
                model_id = $ModelId
                language_code = 'en'
                voice_settings = $voiceSettings
            }
            if ($Seed -gt 0) { $body['seed'] = $Seed }
            Invoke-WebRequest -Method POST -Uri "$apiBase/text-to-speech/${VoiceId}?output_format=mp3_44100_128" -Headers $headers -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 8 -Compress) -OutFile $OutputPath -TimeoutSec 120
            Write-JsonResult ([ordered]@{
                action = 'speech_synthesized'
                voice_id = $VoiceId
                model_id = $ModelId
                text = $Text
                charged_characters = $Text.Length
                voice_settings = $body.voice_settings
                output_path = $OutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }

        'SynthesizeWithTimestamps' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($VoiceId) -or [string]::IsNullOrWhiteSpace($Text) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'VoiceId, Text, and OutputPath are required for SynthesizeWithTimestamps.'
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            if ([string]::IsNullOrWhiteSpace($AlignmentOutputPath)) {
                $AlignmentOutputPath = [System.IO.Path]::ChangeExtension($OutputPath, '.alignment.json')
            }
            $alignmentParent = Split-Path -Parent $AlignmentOutputPath
            if (-not [string]::IsNullOrWhiteSpace($alignmentParent)) {
                New-Item -ItemType Directory -Path $alignmentParent -Force | Out-Null
            }

            $voiceSettings = if ($ModelId -like 'eleven_v3*') {
                [ordered]@{ stability = $Stability }
            } else {
                [ordered]@{
                    stability = $Stability
                    similarity_boost = $SimilarityBoost
                    style = $Style
                    use_speaker_boost = [bool]$UseSpeakerBoost
                    speed = $Speed
                }
            }
            $body = [ordered]@{
                text = $Text
                model_id = $ModelId
                language_code = 'en'
                voice_settings = $voiceSettings
            }
            if ($Seed -gt 0) { $body['seed'] = $Seed }
            # Starter exposes 128 kbps MP3 for the timestamped route. The film
            # pipeline decodes this once to 48 kHz / 24-bit PCM before editing.
            $response = Invoke-ElevenLabsJson -Method POST -Path "/text-to-speech/${VoiceId}/with-timestamps?output_format=mp3_44100_128" -Body $body
            [System.IO.File]::WriteAllBytes($OutputPath, [Convert]::FromBase64String([string]$response.audio_base64))

            $alignmentRecord = [ordered]@{
                voice_id = $VoiceId
                model_id = $ModelId
                text = $Text
                voice_settings = $body.voice_settings
                alignment = $response.alignment
                normalized_alignment = $response.normalized_alignment
            }
            $alignmentRecord | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $AlignmentOutputPath -Encoding utf8

            Write-JsonResult ([ordered]@{
                action = 'speech_synthesized_with_timestamps'
                voice_id = $VoiceId
                model_id = $ModelId
                text = $Text
                charged_characters = $Text.Length
                voice_settings = $body.voice_settings
                output_path = $OutputPath
                alignment_path = $AlignmentOutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
                alignment_sha256 = (Get-FileHash -LiteralPath $AlignmentOutputPath -Algorithm SHA256).Hash
            })
        }

        'Dialogue' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($DialogueInputPath) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'DialogueInputPath and OutputPath are required for Dialogue.'
            }
            if (-not (Test-Path -LiteralPath $DialogueInputPath -PathType Leaf)) {
                throw "Dialogue input file was not found: $DialogueInputPath"
            }

            $dialogueDocument = Get-Content -LiteralPath $DialogueInputPath -Raw | ConvertFrom-Json
            $dialogueInputs = @($dialogueDocument.inputs)
            if ($dialogueInputs.Count -lt 2) {
                throw 'Dialogue input must contain at least two turns.'
            }
            foreach ($turn in $dialogueInputs) {
                if ([string]::IsNullOrWhiteSpace([string]$turn.text) -or [string]::IsNullOrWhiteSpace([string]$turn.voice_id)) {
                    throw 'Every dialogue turn must contain non-empty text and voice_id values.'
                }
            }

            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            if ([string]::IsNullOrWhiteSpace($AlignmentOutputPath)) {
                $AlignmentOutputPath = [System.IO.Path]::ChangeExtension($OutputPath, '.alignment.json')
            }
            $alignmentParent = Split-Path -Parent $AlignmentOutputPath
            if (-not [string]::IsNullOrWhiteSpace($alignmentParent)) {
                New-Item -ItemType Directory -Path $alignmentParent -Force | Out-Null
            }

            $body = [ordered]@{
                inputs = @($dialogueInputs | ForEach-Object {
                    [ordered]@{ text = [string]$_.text; voice_id = [string]$_.voice_id }
                })
                model_id = 'eleven_v3'
                language_code = 'en'
                seed = $Seed
                apply_text_normalization = 'auto'
            }
            $response = Invoke-ElevenLabsJson -Method POST -Path '/text-to-dialogue/with-timestamps?output_format=mp3_44100_128' -Body $body
            [System.IO.File]::WriteAllBytes($OutputPath, [Convert]::FromBase64String([string]$response.audio_base64))

            $alignmentRecord = [ordered]@{
                model_id = $body.model_id
                language_code = $body.language_code
                seed = $Seed
                inputs = $body.inputs
                voice_segments = $response.voice_segments
                alignment = $response.alignment
                normalized_alignment = $response.normalized_alignment
            }
            $alignmentRecord | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $AlignmentOutputPath -Encoding utf8

            $chargedCharacters = ($dialogueInputs | ForEach-Object { ([string]$_.text).Length } | Measure-Object -Sum).Sum
            Write-JsonResult ([ordered]@{
                action = 'dialogue_synthesized'
                model_id = $body.model_id
                seed = $Seed
                turns = $dialogueInputs.Count
                charged_characters = $chargedCharacters
                output_path = $OutputPath
                alignment_path = $AlignmentOutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }

        'SpeechToSpeech' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($VoiceId) -or [string]::IsNullOrWhiteSpace($InputAudioPath) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'VoiceId, InputAudioPath, and OutputPath are required for SpeechToSpeech.'
            }
            if (-not (Test-Path -LiteralPath $InputAudioPath -PathType Leaf)) {
                throw "Speech-to-speech performance input was not found: $InputAudioPath"
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'audio/mpeg' }
            $voiceSettings = [ordered]@{
                stability = $Stability
                similarity_boost = $SimilarityBoost
                style = 0.0
                use_speaker_boost = [bool]$UseSpeakerBoost
            } | ConvertTo-Json -Compress
            $form = @{
                audio = Get-Item -LiteralPath $InputAudioPath
                model_id = $StsModelId
                voice_settings = $voiceSettings
                remove_background_noise = ([bool]$RemoveBackgroundNoise).ToString().ToLowerInvariant()
                file_format = 'other'
            }
            if ($Seed -gt 0) { $form['seed'] = [string]$Seed }
            if ($StsOutputFormat -eq 'pcm_48000') {
                $temporaryPcmPath = $OutputPath + '.tmp.pcm'
                try {
                    Invoke-WebRequest -Method POST -Uri "$apiBase/speech-to-speech/${VoiceId}?output_format=$StsOutputFormat" -Headers $headers -Form $form -OutFile $temporaryPcmPath -TimeoutSec 300
                    & ffmpeg -hide_banner -loglevel error -f s16le -ar 48000 -ac 1 -i $temporaryPcmPath -c:a pcm_s16le -y $OutputPath
                    if ($LASTEXITCODE -ne 0) {
                        throw "FFmpeg could not wrap the 48 kHz PCM response as WAV (exit code $LASTEXITCODE)."
                    }
                }
                finally {
                    if (Test-Path -LiteralPath $temporaryPcmPath -PathType Leaf) {
                        Remove-Item -LiteralPath $temporaryPcmPath -Force
                    }
                }
            }
            else {
                Invoke-WebRequest -Method POST -Uri "$apiBase/speech-to-speech/${VoiceId}?output_format=$StsOutputFormat" -Headers $headers -Form $form -OutFile $OutputPath -TimeoutSec 300
            }
            Write-JsonResult ([ordered]@{
                action = 'speech_to_speech_converted'
                voice_id = $VoiceId
                model_id = $StsModelId
                output_format = $StsOutputFormat
                input_audio_path = $InputAudioPath
                remove_background_noise = [bool]$RemoveBackgroundNoise
                output_path = $OutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }

        'SoundEffect' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($Text) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'Text and OutputPath are required for SoundEffect.'
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'audio/mpeg' }
            $body = [ordered]@{
                text = $Text
                loop = [bool]$LoopSound
                duration_seconds = $DurationSeconds
                prompt_influence = $PromptInfluence
                model_id = 'eleven_text_to_sound_v2'
            }
            $response = Invoke-WebRequest -Method POST -Uri "$apiBase/sound-generation?output_format=mp3_44100_128" -Headers $headers -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 8 -Compress) -OutFile $OutputPath -TimeoutSec 180
            $responseHeaders = @{}
            if ($null -ne $response -and $null -ne $response.PSObject.Properties['Headers']) {
                $responseHeaders = $response.Headers
            }
            Write-JsonResult ([ordered]@{
                action = 'sound_effect_synthesized'
                model_id = $body.model_id
                text = $Text
                duration_seconds = $DurationSeconds
                loop = [bool]$LoopSound
                prompt_influence = $PromptInfluence
                character_cost = $responseHeaders['character-cost']
                request_id = $responseHeaders['request-id']
                output_path = $OutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }

        'Music' {
            Assert-CreditSpendConfirmed
            if ([string]::IsNullOrWhiteSpace($Text) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'Text and OutputPath are required for Music.'
            }
            $parent = Split-Path -Parent $OutputPath
            if (-not [string]::IsNullOrWhiteSpace($parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            $headers = @{ 'xi-api-key' = $script:apiKey; Accept = 'audio/mpeg' }
            $body = [ordered]@{
                prompt = $Text
                music_length_ms = $MusicLengthMs
            }
            $response = Invoke-WebRequest -Method POST -Uri "$apiBase/music?output_format=mp3_44100_128" -Headers $headers -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 8 -Compress) -OutFile $OutputPath -TimeoutSec 300
            $responseHeaders = @{}
            if ($null -ne $response -and $null -ne $response.PSObject.Properties['Headers']) {
                $responseHeaders = $response.Headers
            }
            Write-JsonResult ([ordered]@{
                action = 'music_composed'
                prompt = $Text
                music_length_ms = $MusicLengthMs
                character_cost = $responseHeaders['character-cost']
                request_id = $responseHeaders['request-id']
                output_path = $OutputPath
                sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
            })
        }
    }
}
finally {
    $script:apiKey = $null
    [System.GC]::Collect()
}
