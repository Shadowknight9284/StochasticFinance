# Simple LaTeX Build Script for Strategy Papers
# ============================================

param([string]$TexFile = "")

function Build-LatexPaper {
    param([string]$FilePath)
    
    $Dir = Split-Path $FilePath -Parent
    $Name = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
    $OriginalDir = Get-Location
    
    Write-Host "Building: $FilePath" -ForegroundColor Green
    Set-Location $Dir
    
    # Compile LaTeX (3 passes for references)
    Write-Host "Pass 1..." -ForegroundColor Yellow
    $result1 = pdflatex -interaction=nonstopmode "$Name.tex" 2>&1
    
    Write-Host "Pass 2..." -ForegroundColor Yellow  
    $result2 = pdflatex -interaction=nonstopmode "$Name.tex" 2>&1
    
    Write-Host "Pass 3..." -ForegroundColor Yellow
    $result3 = pdflatex -interaction=nonstopmode "$Name.tex" 2>&1
    
    if (Test-Path "$Name.pdf") {
        Write-Host "SUCCESS: $Name.pdf created" -ForegroundColor Green
        
        # Clean up auxiliary files
        $patterns = @("*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", 
                     "*.run.xml", "*.bcf", "*.fdb_latexmk", "*.fls", "*.synctex.gz")
        
        foreach ($pattern in $patterns) {
            Remove-Item $pattern -ErrorAction SilentlyContinue
        }
        Write-Host "Cleaned auxiliary files" -ForegroundColor Cyan
    } else {
        Write-Host "ERROR: PDF generation failed" -ForegroundColor Red
        Write-Host "Last error output:" -ForegroundColor Yellow
        Write-Host ($result3 | Select-Object -Last 10) -ForegroundColor Red
    }
    
    Set-Location $OriginalDir
}

# Main execution
if ($TexFile -eq "") {
    # Build all papers
    Write-Host "Building all strategy papers..." -ForegroundColor Blue
    $papers = Get-ChildItem -Path "strategies\*\paper.tex"
    foreach ($paper in $papers) {
        Build-LatexPaper $paper.FullName
        Write-Host ""
    }
} else {
    # Build specific paper
    if (Test-Path $TexFile) {
        Build-LatexPaper $TexFile
    } else {
        Write-Host "File not found: $TexFile" -ForegroundColor Red
    }
}

Write-Host "Build process complete!" -ForegroundColor Green
