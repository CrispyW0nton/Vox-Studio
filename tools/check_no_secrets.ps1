[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string[]]$AdditionalPaths = @(),
    [switch]$ScanHistory
)

$ErrorActionPreference = 'Stop'

$forbiddenPathPatterns = @(
    '(?i)(^|/)\.env(?:\.|$)',
    '(?i)(^|/)elevenlabs\.bin$',
    '(?i)(^|/)(?:credentials?|api[-_]?keys?)\.(?:json|txt|ini)$'
)

$secretPatterns = @(
    @{
        Name = 'ElevenLabs-style API key'
        Pattern = '(?i)(?<![A-Za-z0-9])sk_[A-Za-z0-9_-]{24,}'
    },
    @{
        Name = 'assigned ElevenLabs API key'
        Pattern = '(?im)^\s*(?:ELEVENLABS_API_KEY|XI_API_KEY)\s*[:=]\s*["'']?(?!<|YOUR_|REPLACE_|EXAMPLE_|TEST_)[A-Za-z0-9_-]{16,}'
    }
)

$violations = [System.Collections.Generic.List[string]]::new()

function Test-ForbiddenPath {
    param(
        [Parameter(Mandatory)]
        [string]$RelativePath,
        [string]$DisplayPrefix = ''
    )

    $normalizedPath = $RelativePath.Replace('\', '/')
    foreach ($pattern in $forbiddenPathPatterns) {
        if ($normalizedPath -match $pattern) {
            $violations.Add(
                "Forbidden credential path: $DisplayPrefix$normalizedPath"
            )
        }
    }
}

function Test-FileForSecrets {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$DisplayPath
    )

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $decodedValues = @(
        [System.Text.Encoding]::UTF8.GetString($bytes),
        [System.Text.Encoding]::Unicode.GetString($bytes),
        [System.Text.Encoding]::BigEndianUnicode.GetString($bytes)
    )

    foreach ($secretPattern in $secretPatterns) {
        if ($decodedValues | Where-Object { $_ -match $secretPattern.Pattern } |
            Select-Object -First 1) {
            $violations.Add("$($secretPattern.Name) detected in: $DisplayPath")
        }
    }
}

function Test-TextForSecrets {
    param(
        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Text,
        [Parameter(Mandatory)]
        [string]$DisplayPath
    )

    foreach ($secretPattern in $secretPatterns) {
        if ($Text -match $secretPattern.Pattern) {
            $violations.Add("$($secretPattern.Name) detected in: $DisplayPath")
        }
    }
}

function Test-DirectoryForSecrets {
    param(
        [Parameter(Mandatory)]
        [string]$Root,
        [Parameter(Mandatory)]
        [string]$DisplayPrefix,
        [int]$ArchiveDepth = 0
    )

    foreach ($file in Get-ChildItem -LiteralPath $Root -File -Recurse) {
        $relativePath = [System.IO.Path]::GetRelativePath($Root, $file.FullName)
        Test-ForbiddenPath -RelativePath $relativePath -DisplayPrefix $DisplayPrefix
        Test-FileForSecrets -Path $file.FullName `
            -DisplayPath "$DisplayPrefix$($relativePath.Replace('\', '/'))"

        if ($file.Extension -ieq '.zip' -and $ArchiveDepth -lt 2) {
            $archiveRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
                'voxstudio_archive_scan_' + [System.Guid]::NewGuid().ToString('N')
            )
            try {
                Expand-Archive -LiteralPath $file.FullName -DestinationPath $archiveRoot
                Test-DirectoryForSecrets -Root $archiveRoot `
                    -DisplayPrefix "$DisplayPrefix$relativePath!/" `
                    -ArchiveDepth ($ArchiveDepth + 1)
            } finally {
                Remove-Item -LiteralPath $archiveRoot -Recurse -Force `
                    -ErrorAction SilentlyContinue
            }
        }
    }
}

Push-Location -LiteralPath $RepositoryRoot
try {
    $trackedFiles = @(git ls-files)
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to list tracked repository files.'
    }

    foreach ($relativePath in $trackedFiles) {
        Test-ForbiddenPath -RelativePath $relativePath
        Test-FileForSecrets -Path (Join-Path $RepositoryRoot $relativePath) `
            -DisplayPath $relativePath.Replace('\', '/')
    }

    if ($ScanHistory) {
        $historyRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
            'voxstudio_history_scan_' + [System.Guid]::NewGuid().ToString('N')
        )
        New-Item -ItemType Directory -Path $historyRoot | Out-Null
        try {
            $commits = @(git rev-list --all)
            if ($LASTEXITCODE -ne 0) {
                throw 'Unable to list repository history.'
            }
            foreach ($commit in $commits) {
                $archivePath = Join-Path $historyRoot "$commit.zip"
                $treePath = Join-Path $historyRoot $commit
                git archive --format=zip "--output=$archivePath" $commit
                if ($LASTEXITCODE -ne 0) {
                    throw "Unable to inspect repository commit $commit."
                }
                Expand-Archive -LiteralPath $archivePath -DestinationPath $treePath
                Test-DirectoryForSecrets -Root $treePath `
                    -DisplayPrefix "history:$($commit.Substring(0, 12)):"
            }

            $commitMessages = (git log --all --format='%B') -join "`n"
            if ($LASTEXITCODE -ne 0) {
                throw 'Unable to inspect commit messages.'
            }
            Test-TextForSecrets -Text $commitMessages `
                -DisplayPath 'repository commit messages'

            $tagMessages = (git for-each-ref refs/tags --format='%(contents)') -join "`n"
            if ($LASTEXITCODE -ne 0) {
                throw 'Unable to inspect tag messages.'
            }
            Test-TextForSecrets -Text $tagMessages `
                -DisplayPath 'repository tag messages'
        } finally {
            Remove-Item -LiteralPath $historyRoot -Recurse -Force `
                -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}

foreach ($additionalPath in $AdditionalPaths) {
    $resolvedPath = (Resolve-Path -LiteralPath $additionalPath).Path
    if ([System.IO.File]::Exists($resolvedPath)) {
        $name = [System.IO.Path]::GetFileName($resolvedPath)
        Test-ForbiddenPath -RelativePath $name -DisplayPrefix 'release:'
        Test-FileForSecrets -Path $resolvedPath -DisplayPath "release:$name"
        if ([System.IO.Path]::GetExtension($resolvedPath) -ieq '.zip') {
            $archiveRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
                'voxstudio_release_scan_' + [System.Guid]::NewGuid().ToString('N')
            )
            try {
                Expand-Archive -LiteralPath $resolvedPath -DestinationPath $archiveRoot
                Test-DirectoryForSecrets -Root $archiveRoot `
                    -DisplayPrefix "release:$name!/"
            } finally {
                Remove-Item -LiteralPath $archiveRoot -Recurse -Force `
                    -ErrorAction SilentlyContinue
            }
        }
        continue
    }

    Test-DirectoryForSecrets -Root $resolvedPath -DisplayPrefix 'release:'
}

if ($violations.Count -gt 0) {
    $violations | Sort-Object -Unique | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Host 'No bundled ElevenLabs credentials found.'
