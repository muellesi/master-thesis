
param(
    [switch] $u
)

$anaconda_path = "C:\Program Files\Anaconda3"

if (Test-Path "$env:USERPROFILE\Anaconda3" -PathType Container) {
    $anaconda_path = "$env:USERPROFILE\Anaconda3"
    Write-Output "Found Anaconda user installation!"
}
if (Test-Path "$env:LOCALAPPDATA\Continuum\anaconda3" -PathType Container) {
    $anaconda_path = "$env:LOCALAPPDATA\Continuum\anaconda3"
    Write-Output "Found Anaconda user installation!"
}
elseif (Test-Path "$env:USERPROFILE\Miniconda3" -PathType Container) {
    $anaconda_path = "$env:USERPROFILE\Miniconda3"
    Write-Output "Found Miniconda user installation!"
}
elseif (Test-Path "C:\Program Files\Miniconda3" -PathType Container) {
    $anaconda_path = "C:\Program Files\Miniconda3"
    Write-Output "Found Miniconda global installation!"
}
elseif (Test-Path "C:\tools\Anaconda3" -PathType Container) {
    $anaconda_path = "C:\tools\Anaconda3"
    Write-Output "Found Anaconda ITIV installation!"
}

if (!(Test-Path $anaconda_path -PathType Container)) {
    Write-Output "No valid conda installation was found!"
} else {
    & "$anaconda_path\shell\condabin\conda-hook.ps1"
    conda activate "$anaconda_path"

    $env_name = ""
    Get-Content -Path environment.yml | foreach-object {
        if ($_.StartsWith("name:")){
            Write-Output $_
            $env_name = $_.split(":")[1]
            $env_name = $env_name.Trim()
        }
    }

    if ($env_name -ne "") {
        if ($u) {
            Write-Output "Updating Anaconda env $env_name..."
            conda env update -f .\environment.yml
        }
    
        Write-Output "Activating Anaconda environment $env_name..."
        conda activate $env_name
    } else {
        Write-Output "Could not retrieve environment name from environment.yml!"
    }
}


