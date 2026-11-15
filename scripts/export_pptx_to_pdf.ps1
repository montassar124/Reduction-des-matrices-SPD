param(
    [Parameter(Mandatory=$true)][string]$pptxPath,
    [Parameter(Mandatory=$true)][string]$pdfPath
)

$ppt = New-Object -ComObject PowerPoint.Application
try {
    $presentation = $ppt.Presentations.Open($pptxPath, 1, 0, 0)
    $presentation.SaveAs($pdfPath, 32)
    $presentation.Close()
    Write-Host "Successfully exported $pptxPath to $pdfPath"
} catch {
    Write-Error $_.Exception.Message
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
