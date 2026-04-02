param(
    [string]$RootPath = (Join-Path $PSScriptRoot "..\..\data\raw\aihub"),
    [string[]]$DatasetNames = @(),
    [int]$MaxArchives = 0,
    [switch]$Overwrite,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [string]$Tag,
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Tag] $Message"
}

function Get-NormalizedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return [System.IO.Path]::GetFullPath($Path)
}

function Test-PathWithinRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [string]$RootPath
    )

    $normalizedRoot = Get-NormalizedPath -Path $RootPath
    $normalizedTarget = Get-NormalizedPath -Path $TargetPath

    if (-not $normalizedRoot.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $normalizedRoot += [System.IO.Path]::DirectorySeparatorChar
    }

    return $normalizedTarget.StartsWith(
        $normalizedRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Remove-SafeDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [string]$RootPath
    )

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        return
    }

    if (-not (Test-PathWithinRoot -TargetPath $TargetPath -RootPath $RootPath)) {
        throw "삭제 대상이 허용된 루트 밖에 있습니다: $TargetPath"
    }

    Remove-Item -LiteralPath $TargetPath -Recurse -Force
}

function Test-DatasetMatch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetName,
        [string[]]$Patterns
    )

    if ($Patterns.Count -eq 0) {
        return $true
    }

    foreach ($pattern in $Patterns) {
        if ($DatasetName -like $pattern) {
            return $true
        }
    }

    return $false
}

function Get-ArchiveMarkerPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    return Join-Path $DestinationPath ".extract_complete.json"
}

function Get-ArchiveState {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$ZipFile,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    $markerPath = Get-ArchiveMarkerPath -DestinationPath $DestinationPath
    if (-not (Test-Path -LiteralPath $markerPath)) {
        return $null
    }

    try {
        $raw = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8
        $state = $raw | ConvertFrom-Json
    }
    catch {
        return $null
    }

    if (
        $state.zip_name -eq $ZipFile.Name -and
        [int64]$state.zip_size -eq $ZipFile.Length -and
        $state.zip_last_write_utc -eq $ZipFile.LastWriteTimeUtc.ToString("o")
    ) {
        return $state
    }

    return $null
}

function Write-ArchiveState {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$ZipFile,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [Parameter(Mandatory = $true)]
        [string]$DatasetName,
        [Parameter(Mandatory = $true)]
        [TimeSpan]$Elapsed
    )

    $markerPath = Get-ArchiveMarkerPath -DestinationPath $DestinationPath
    $fileCount = @(Get-ChildItem -LiteralPath $DestinationPath -Recurse -File -Force).Count
    $state = [ordered]@{
        zip_name = $ZipFile.Name
        zip_path = $ZipFile.FullName
        zip_size = $ZipFile.Length
        zip_last_write_utc = $ZipFile.LastWriteTimeUtc.ToString("o")
        dataset_name = $DatasetName
        extracted_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        elapsed_seconds = [math]::Round($Elapsed.TotalSeconds, 2)
        extracted_file_count = $fileCount
    }

    $json = $state | ConvertTo-Json -Depth 4
    Set-Content -LiteralPath $markerPath -Value $json -Encoding UTF8
}

function Invoke-ArchiveExtraction {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$ZipFile,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [Parameter(Mandatory = $true)]
        [string]$TarPath,
        [Parameter(Mandatory = $true)]
        [string]$RootPath,
        [Parameter(Mandatory = $true)]
        [string]$DatasetName,
        [switch]$Overwrite,
        [switch]$DryRun
    )

    $parentDirectory = Split-Path -Path $DestinationPath -Parent
    $tempDirectoryName = "{0}.__extracting__" -f [System.IO.Path]::GetFileName($DestinationPath)
    $tempDirectory = Join-Path $parentDirectory $tempDirectoryName

    if ($DryRun) {
        Write-Log -Tag "DRYRUN" -Message ("{0} -> {1}" -f $ZipFile.FullName, $DestinationPath)
        return "dryrun"
    }

    Remove-SafeDirectory -TargetPath $tempDirectory -RootPath $RootPath
    New-Item -ItemType Directory -Path $tempDirectory | Out-Null

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        # 각 zip을 동명 폴더로 분리해 풀어야 파일명 충돌과 재실행 시 상태 추적이 쉬워진다.
        & $TarPath -xf $ZipFile.FullName -C $tempDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "tar.exe 종료 코드가 0이 아닙니다: $LASTEXITCODE"
        }

        if (Test-Path -LiteralPath $DestinationPath) {
            if (-not $Overwrite) {
                throw "대상 폴더가 이미 존재합니다: $DestinationPath"
            }

            Remove-SafeDirectory -TargetPath $DestinationPath -RootPath $RootPath
        }

        Move-Item -LiteralPath $tempDirectory -Destination $DestinationPath
        $stopwatch.Stop()
        Write-ArchiveState `
            -ZipFile $ZipFile `
            -DestinationPath $DestinationPath `
            -DatasetName $DatasetName `
            -Elapsed $stopwatch.Elapsed

        $state = Get-ArchiveState -ZipFile $ZipFile -DestinationPath $DestinationPath
        Write-Log `
            -Tag "DONE" `
            -Message ("dataset={0} zip={1} files={2} elapsed={3}s" -f
                $DatasetName,
                $ZipFile.Name,
                $state.extracted_file_count,
                $state.elapsed_seconds
            )
        return "done"
    }
    catch {
        $stopwatch.Stop()
        Remove-SafeDirectory -TargetPath $tempDirectory -RootPath $RootPath
        throw
    }
}

function Get-ArchiveCommand {
    # Windows에서는 tar.exe가 흔하고, macOS/Linux에서는 tar가 일반적이다.
    # 둘 중 현재 환경에서 실제로 존재하는 실행 파일을 찾아 같은 추출 로직을 재사용한다.
    foreach ($commandName in @("tar.exe", "tar")) {
        try {
            return Get-Command $commandName -ErrorAction Stop
        }
        catch {
            continue
        }
    }

    throw "tar 실행 파일을 찾을 수 없습니다. tar 또는 tar.exe가 PATH에 있어야 합니다."
}

$resolvedRoot = Get-NormalizedPath -Path $RootPath
if (-not (Test-Path -LiteralPath $resolvedRoot)) {
    throw "AI Hub 루트를 찾을 수 없습니다: $resolvedRoot"
}

$archives = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -Filter *.zip -File | Sort-Object FullName
Write-Log -Tag "START" -Message ("root={0}" -f $resolvedRoot)
Write-Log -Tag "START" -Message ("archives={0}" -f @($archives).Count)

if (@($archives).Count -eq 0) {
    Write-Log `
        -Tag "INFO" `
        -Message "root 아래에서 추출할 zip 파일을 찾지 못했습니다. *.zip 파일을 넣고 다시 실행하세요."
    Write-Log -Tag "SUMMARY" -Message "done=0 skipped=0 dryrun=0 failed=0"
    return
}

$tarCommand = Get-ArchiveCommand

$plannedArchives = @()
foreach ($archive in $archives) {
    $relativePath = $archive.FullName.Substring($resolvedRoot.Length).TrimStart([char[]]@('\', '/'))
    $relativeSegments = $relativePath -split "[\\/]"
    if ($relativeSegments.Count -lt 2) {
        continue
    }

    $datasetName = $relativeSegments[0]
    if (-not (Test-DatasetMatch -DatasetName $datasetName -Patterns $DatasetNames)) {
        continue
    }

    $destinationPath = Join-Path $archive.DirectoryName $archive.BaseName
    $plannedArchives += [PSCustomObject]@{
        ZipFile = $archive
        DatasetName = $datasetName
        DestinationPath = $destinationPath
    }
}

if ($MaxArchives -gt 0) {
    $plannedArchives = @($plannedArchives | Select-Object -First $MaxArchives)
}

Write-Log -Tag "PLAN" -Message ("selected_archives={0}" -f @($plannedArchives).Count)
if ($DatasetNames.Count -gt 0) {
    Write-Log -Tag "FILTER" -Message ("dataset_patterns={0}" -f ($DatasetNames -join ", "))
}

$doneCount = 0
$skippedCount = 0
$dryRunCount = 0
$failedCount = 0

foreach ($item in $plannedArchives) {
    $zipFile = $item.ZipFile
    $destinationPath = $item.DestinationPath
    $datasetName = $item.DatasetName

    $existingState = Get-ArchiveState -ZipFile $zipFile -DestinationPath $destinationPath
    if ($existingState -and -not $Overwrite) {
        $skippedCount += 1
        Write-Log `
            -Tag "SKIP" `
            -Message ("dataset={0} zip={1} reason=already_extracted files={2}" -f
                $datasetName,
                $zipFile.Name,
                $existingState.extracted_file_count
            )
        continue
    }

    Write-Log `
        -Tag "RUN" `
        -Message ("dataset={0} zip={1}" -f $datasetName, $zipFile.Name)

    try {
        $extractParams = @{
            ZipFile = $zipFile
            DestinationPath = $destinationPath
            TarPath = $tarCommand.Source
            RootPath = $resolvedRoot
            DatasetName = $datasetName
        }
        if ($Overwrite.IsPresent) {
            $extractParams.Overwrite = $true
        }
        if ($DryRun.IsPresent) {
            $extractParams.DryRun = $true
        }

        $result = Invoke-ArchiveExtraction @extractParams

        switch ($result) {
            "done" { $doneCount += 1 }
            "dryrun" { $dryRunCount += 1 }
            default { }
        }
    }
    catch {
        $failedCount += 1
        Write-Log `
            -Tag "FAIL" `
            -Message ("dataset={0} zip={1} error={2}" -f
                $datasetName,
                $zipFile.Name,
                $_.Exception.Message
            )
    }
}

Write-Log `
    -Tag "SUMMARY" `
    -Message ("done={0} skipped={1} dryrun={2} failed={3}" -f
        $doneCount,
        $skippedCount,
        $dryRunCount,
        $failedCount
    )
