[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ScannerPath
)

$ErrorActionPreference = 'Stop'
$scanner = (Resolve-Path -LiteralPath $ScannerPath).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'voxstudio_secret_scanner_' + [System.Guid]::NewGuid().ToString('N')
)

function New-TestRepository {
    param([string]$Name)

    $path = Join-Path $testRoot $Name
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    git -C $path init --quiet
    git -C $path config user.email 'tests@voxstudio.local'
    git -C $path config user.name 'Vox Studio Tests'
    return $path
}

function Invoke-Scanner {
    param(
        [string]$Repository,
        [string[]]$Arguments = @()
    )

    & (Join-Path $PSHOME 'pwsh.exe') -NoProfile -File $scanner `
        -RepositoryRoot $Repository @Arguments *> $null
    return $LASTEXITCODE
}

try {
    $cleanRepository = New-TestRepository 'clean'
    [System.IO.File]::WriteAllText(
        (Join-Path $cleanRepository 'settings.txt'),
        'ELEVENLABS_API_KEY=YOUR_KEY_HERE'
    )
    git -C $cleanRepository add settings.txt
    if ((Invoke-Scanner $cleanRepository) -ne 0) {
        throw 'Clean placeholders should pass the credential scanner.'
    }

    $asciiRepository = New-TestRepository 'ascii-secret'
    [System.IO.File]::WriteAllText(
        (Join-Path $asciiRepository 'defaults.cfg'),
        ('key=sk_' + ('A' * 32))
    )
    git -C $asciiRepository add defaults.cfg
    if ((Invoke-Scanner $asciiRepository) -eq 0) {
        throw 'ASCII credentials must fail the credential scanner.'
    }

    $utf16Repository = New-TestRepository 'utf16-secret'
    [System.IO.File]::WriteAllText(
        (Join-Path $utf16Repository 'defaults.dat'),
        ('key=sk_' + ('B' * 32)),
        [System.Text.Encoding]::Unicode
    )
    git -C $utf16Repository add defaults.dat
    if ((Invoke-Scanner $utf16Repository) -eq 0) {
        throw 'UTF-16 credentials must fail the credential scanner.'
    }

    $historyRepository = New-TestRepository 'history-secret'
    [System.IO.File]::WriteAllText(
        (Join-Path $historyRepository 'removed.txt'),
        ('sk_' + ('C' * 32)),
        [System.Text.Encoding]::Unicode
    )
    git -C $historyRepository add removed.txt
    git -C $historyRepository commit --quiet -m 'Add fixture'
    Remove-Item -LiteralPath (Join-Path $historyRepository 'removed.txt')
    git -C $historyRepository add --all
    git -C $historyRepository commit --quiet -m 'Remove fixture'
    if ((Invoke-Scanner $historyRepository @('-ScanHistory')) -eq 0) {
        throw 'Credentials in Git history must fail the credential scanner.'
    }

    $messageRepository = New-TestRepository 'message-secret'
    [System.IO.File]::WriteAllText(
        (Join-Path $messageRepository 'clean.txt'),
        'clean'
    )
    git -C $messageRepository add clean.txt
    git -C $messageRepository commit --quiet `
        -m ('Accidentally included sk_' + ('F' * 32))
    if ((Invoke-Scanner $messageRepository @('-ScanHistory')) -eq 0) {
        throw 'Credentials in commit messages must fail the credential scanner.'
    }

    $releaseRepository = New-TestRepository 'release-secret'
    [System.IO.File]::WriteAllText(
        (Join-Path $releaseRepository 'clean.txt'),
        'clean'
    )
    git -C $releaseRepository add clean.txt
    $releasePath = Join-Path $releaseRepository 'release'
    New-Item -ItemType Directory -Path $releasePath | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $releasePath 'runtime.cfg'),
        ('sk_' + ('D' * 32))
    )
    if ((Invoke-Scanner $releaseRepository @('-AdditionalPaths', $releasePath)) -eq 0) {
        throw 'Credentials in release payloads must fail the credential scanner.'
    }

    $releasePathRepository = New-TestRepository 'release-path'
    [System.IO.File]::WriteAllText(
        (Join-Path $releasePathRepository 'clean.txt'),
        'clean'
    )
    git -C $releasePathRepository add clean.txt
    $forbiddenReleasePath =
        Join-Path $releasePathRepository 'release\secrets'
    New-Item -ItemType Directory -Path $forbiddenReleasePath -Force | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $forbiddenReleasePath 'elevenlabs.bin'),
        'encrypted-looking payload'
    )
    if ((Invoke-Scanner $releasePathRepository @(
        '-AdditionalPaths',
        (Join-Path $releasePathRepository 'release')
    )) -eq 0) {
        throw 'Forbidden credential paths in release payloads must fail the scanner.'
    }

    $zipRepository = New-TestRepository 'release-zip'
    [System.IO.File]::WriteAllText((Join-Path $zipRepository 'clean.txt'), 'clean')
    git -C $zipRepository add clean.txt
    $zipContents = Join-Path $zipRepository 'zip-contents'
    New-Item -ItemType Directory -Path $zipContents | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $zipContents 'runtime.cfg'),
        ('sk_' + ('E' * 32))
    )
    $zipPath = Join-Path $zipRepository 'VoxStudio.zip'
    Compress-Archive -Path (Join-Path $zipContents '*') -DestinationPath $zipPath
    if ((Invoke-Scanner $zipRepository @('-AdditionalPaths', $zipPath)) -eq 0) {
        throw 'Credentials inside release ZIP files must fail the scanner.'
    }
} finally {
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host 'Secret scanner rejection tests passed.'
exit 0
