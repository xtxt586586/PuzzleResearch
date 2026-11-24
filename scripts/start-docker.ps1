param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker\compose.yaml"
$envFile = Join-Path $projectRoot ".env"
$secretDirectory = Join-Path $projectRoot ".docker-secrets"
$passwordFile = Join-Path $secretDirectory "postgres_password"

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "Missing project .env. Copy .env.example and configure the model API key first."
}

if (-not (Test-Path -LiteralPath $passwordFile -PathType Leaf)) {
    New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null

    $password = $null
    $existingContainer = docker inspect puzzleresearch-postgres-1 2>$null
    if ($LASTEXITCODE -eq 0) {
        $containerEnvironment = docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' puzzleresearch-postgres-1
        $passwordEntry = $containerEnvironment | Where-Object { $_ -like "POSTGRES_PASSWORD=*" } | Select-Object -First 1
        if ($passwordEntry) {
            $password = $passwordEntry.Substring("POSTGRES_PASSWORD=".Length)
        }
    }

    if (-not $password) {
        $existingVolume = docker volume inspect puzzleresearch_postgres_data 2>$null
        if ($LASTEXITCODE -eq 0) {
            throw "An existing PostgreSQL data volume was found, but its password is unavailable. Restore POSTGRES_PASSWORD before continuing."
        }

        $randomBytes = New-Object byte[] 36
        $randomGenerator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $randomGenerator.GetBytes($randomBytes)
        }
        finally {
            $randomGenerator.Dispose()
        }
        $password = [Convert]::ToBase64String($randomBytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    }

    [System.IO.File]::WriteAllText($passwordFile, $password, [System.Text.Encoding]::ASCII)
}

$env:POSTGRES_PASSWORD = [System.IO.File]::ReadAllText($passwordFile).Trim()
if (-not $env:POSTGRES_PASSWORD) {
    throw "The local PostgreSQL credential file is empty: $passwordFile"
}

Push-Location $projectRoot
try {
    docker compose -f $composeFile up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "PuzzleResearch Docker 服务启动失败，docker compose exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    Remove-Item Env:POSTGRES_PASSWORD -ErrorAction SilentlyContinue
}

Write-Output "PuzzleResearch backend, PostgreSQL, and Redis are running."
