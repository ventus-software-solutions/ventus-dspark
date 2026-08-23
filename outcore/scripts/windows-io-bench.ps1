param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [ValidateRange(1, 10000)]
    [int]$Iterations = 1,
    [ValidateRange(4096, [long]::MaxValue)]
    [long]$Bytes = [long]::MaxValue
)

$ErrorActionPreference = 'Stop'

if (-not ('VentusOutcoreNativeIo' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class VentusOutcoreNativeIo
{
    public const uint GENERIC_READ = 0x80000000;
    public const uint FILE_SHARE_READ = 0x00000001;
    public const uint OPEN_EXISTING = 3;
    public const uint FILE_FLAG_NO_BUFFERING = 0x20000000;
    public const uint FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000;
    public const uint MEM_COMMIT = 0x1000;
    public const uint MEM_RESERVE = 0x2000;
    public const uint MEM_RELEASE = 0x8000;
    public const uint PAGE_READWRITE = 0x04;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateFileW(
        string fileName,
        uint desiredAccess,
        uint shareMode,
        IntPtr securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool ReadFile(
        IntPtr file,
        IntPtr buffer,
        uint bytesToRead,
        out uint bytesRead,
        IntPtr overlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetFilePointerEx(
        IntPtr file,
        long distance,
        out long newPosition,
        uint moveMethod);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr VirtualAlloc(
        IntPtr address,
        UIntPtr size,
        uint allocationType,
        uint protect);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool VirtualFree(IntPtr address, UIntPtr size, uint freeType);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr handle);
}
'@
}

$resolvedPath = (Resolve-Path -LiteralPath $Path).Path
$fileInfo = Get-Item -LiteralPath $resolvedPath
$alignment = 4096L
$chunkBytes = 4MB
$requestedBytes = [Math]::Min($fileInfo.Length, $Bytes)
$readBytes = $requestedBytes - ($requestedBytes % $alignment)
if ($readBytes -le 0) {
    throw 'The aligned read size is zero.'
}

$flags = [VentusOutcoreNativeIo]::FILE_FLAG_NO_BUFFERING -bor [VentusOutcoreNativeIo]::FILE_FLAG_SEQUENTIAL_SCAN
$handle = [VentusOutcoreNativeIo]::CreateFileW(
    $resolvedPath,
    [VentusOutcoreNativeIo]::GENERIC_READ,
    [VentusOutcoreNativeIo]::FILE_SHARE_READ,
    [IntPtr]::Zero,
    [VentusOutcoreNativeIo]::OPEN_EXISTING,
    $flags,
    [IntPtr]::Zero)
if ($handle -eq [IntPtr](-1)) {
    throw "CreateFileW failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
}

$buffer = [VentusOutcoreNativeIo]::VirtualAlloc(
    [IntPtr]::Zero,
    [UIntPtr]$chunkBytes,
    [VentusOutcoreNativeIo]::MEM_COMMIT -bor [VentusOutcoreNativeIo]::MEM_RESERVE,
    [VentusOutcoreNativeIo]::PAGE_READWRITE)
if ($buffer -eq [IntPtr]::Zero) {
    [void][VentusOutcoreNativeIo]::CloseHandle($handle)
    throw "VirtualAlloc failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
}

try {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    for ($iteration = 0; $iteration -lt $Iterations; $iteration++) {
        $newPosition = 0L
        if (-not [VentusOutcoreNativeIo]::SetFilePointerEx($handle, 0, [ref]$newPosition, 0)) {
            throw "SetFilePointerEx failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
        $remaining = $readBytes
        while ($remaining -gt 0) {
            $nextRead = [uint32][Math]::Min($chunkBytes, $remaining)
            $completed = 0
            if (-not [VentusOutcoreNativeIo]::ReadFile($handle, $buffer, $nextRead, [ref]$completed, [IntPtr]::Zero)) {
                throw "ReadFile failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
            }
            if ($completed -ne $nextRead) {
                throw "Short read: expected $nextRead bytes, got $completed."
            }
            $remaining -= $completed
        }
    }
    $timer.Stop()
} finally {
    [void][VentusOutcoreNativeIo]::VirtualFree($buffer, [UIntPtr]::Zero, [VentusOutcoreNativeIo]::MEM_RELEASE)
    [void][VentusOutcoreNativeIo]::CloseHandle($handle)
}

$totalBytes = $readBytes * $Iterations
$gibPerSecond = ($totalBytes / 1GB) / $timer.Elapsed.TotalSeconds
[pscustomobject]@{
    Path = $resolvedPath
    FileBytes = $fileInfo.Length
    ReadBytesPerIteration = $readBytes
    Iterations = $Iterations
    ElapsedSeconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    DirectReadGiBPerSecond = [Math]::Round($gibPerSecond, 3)
}
