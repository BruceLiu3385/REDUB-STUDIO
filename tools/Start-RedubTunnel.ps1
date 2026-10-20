[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9.\-]*$')]
    [string]$SshHost,
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [ValidatePattern('^[a-zA-Z0-9_][a-zA-Z0-9_\-]*$')]
    [string]$SshUser = 'root',
    [ValidateRange(1024, 65535)]
    [int]$LocalPort = 7860,
    [ValidateRange(1024, 65535)]
    [int]$RemotePort = 7860
)

$ErrorActionPreference = 'Stop'
$sshCommand = Get-Command ssh -CommandType Application -ErrorAction Stop
$forward = '127.0.0.1:{0}:127.0.0.1:{1}' -f $LocalPort, $RemotePort
Write-Host ('Open http://127.0.0.1:{0}/ while this SSH session is running.' -f $LocalPort)
Write-Host 'Check the server fingerprint before accepting a first connection. Passwords are not saved by this script.'
& $sshCommand.Source -N -T -p $SshPort -L $forward `
    -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=ask `
    -o ServerAliveInterval=25 -o ServerAliveCountMax=3 `
    ('{0}@{1}' -f $SshUser, $SshHost)
if ($LASTEXITCODE -ne 0) { throw ('SSH tunnel exited with code {0}.' -f $LASTEXITCODE) }
