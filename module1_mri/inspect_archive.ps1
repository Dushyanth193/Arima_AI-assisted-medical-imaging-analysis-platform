Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead('d:\Nexora-hackathon\archive.zip')
$entries = $zip.Entries

Write-Host "Total entries: $($entries.Count)"
Write-Host ""

Write-Host "First 30 entries:"
$entries | Select-Object -First 30 | ForEach-Object {
    Write-Host "$($_.FullName)  ($($_.Length) bytes)"
}

Write-Host ""
Write-Host "Last 10 entries:"
$entries | Select-Object -Last 10 | ForEach-Object {
    Write-Host "$($_.FullName)  ($($_.Length) bytes)"
}

Write-Host ""
Write-Host "File extensions breakdown:"
$entries | ForEach-Object {
    [System.IO.Path]::GetExtension($_.FullName)
} | Group-Object | Sort-Object Count -Descending | Select-Object Count, Name | Format-Table

$zip.Dispose()
