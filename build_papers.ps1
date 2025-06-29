# LaTeX Build Script for Strategy Papers (PowerShell)
# ===================================================
# This script compiles LaTeX papers and cleans up auxiliary files,
# keeping only the final PDF output.

param(
    [string]$TexFile = ""
)

function Compile-Paper {
    param([string]$TexFile)
    
    $PaperDir = Split-Path $TexFile -Parent
    $PaperName = [System.IO.Path]::GetFileNameWithoutExtension($TexFile)
    
    Write-Host "🔨 Compiling: $TexFile" -ForegroundColor Green
    Write-Host "📁 Directory: $PaperDir" -ForegroundColor Cyan
    Write-Host "📄 Output: $PaperName.pdf" -ForegroundColor Yellow
    
    # Change to paper directory
    Push-Location $PaperDir
    
    # First compilation pass
    Write-Host "📝 First pass..." -ForegroundColor Blue
    & pdflatex -interaction=nonstopmode "$PaperName.tex" | Out-Null
    
    # Check if bibliography exists and compile if needed
    if ((Test-Path "$PaperName.bbl") -or (Select-String -Path "$PaperName.tex" -Pattern "\\bibliography" -Quiet)) {
        Write-Host "📚 Processing bibliography..." -ForegroundColor Magenta
        & bibtex "$PaperName" | Out-Null
        & pdflatex -interaction=nonstopmode "$PaperName.tex" | Out-Null
    }
    
    # Final compilation pass
    Write-Host "📝 Final pass..." -ForegroundColor Blue
    & pdflatex -interaction=nonstopmode "$PaperName.tex" | Out-Null
    
    # Check if PDF was generated successfully
    if (Test-Path "$PaperName.pdf") {
        Write-Host "✅ Success: $PaperName.pdf generated" -ForegroundColor Green
        
        # Clean up auxiliary files
        Write-Host "🧹 Cleaning auxiliary files..." -ForegroundColor Yellow
        $AuxFiles = @(
            "*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", 
            "*.run.xml", "*.bcf", "*.fdb_latexmk", "*.fls", 
            "*.synctex.gz", "*.nav", "*.snm", "*.vrb", "*.figlist", 
            "*.makefile", "*.dvi", "*.ps", "*.lof", "*.lot", 
            "*.acn", "*.acr", "*.alg", "*.glg", "*.glo", "*.gls", 
            "*.ist", "*.loa", "*.xdy"
        )
        
        foreach ($Pattern in $AuxFiles) {
            Remove-Item $Pattern -ErrorAction SilentlyContinue
        }
        
        $FullPath = Join-Path (Get-Location) "$PaperName.pdf"
        Write-Host "🎉 Build complete: $FullPath" -ForegroundColor Green
    } else {
        Write-Host "❌ Error: Failed to generate PDF" -ForegroundColor Red
        Write-Host "📋 Check LaTeX errors in log files" -ForegroundColor Yellow
        Pop-Location
        return $false
    }
    
    # Return to original directory
    Pop-Location
    return $true
}

function Compile-AllPapers {
    Write-Host "🚀 Compiling all strategy papers..." -ForegroundColor Green
    
    # Find all paper.tex files in strategies subdirectories
    $TexFiles = Get-ChildItem -Path "strategies\*\paper.tex" -File
    
    foreach ($TexFile in $TexFiles) {
        Compile-Paper $TexFile.FullName
        Write-Host ""
    }
    
    Write-Host "📊 Summary of generated PDFs:" -ForegroundColor Cyan
    $PdfFiles = Get-ChildItem -Path "strategies" -Recurse -Filter "*.pdf"
    foreach ($Pdf in $PdfFiles) {
        Write-Host "  📄 $($Pdf.FullName)" -ForegroundColor White
    }
}

# Main execution
if ([string]::IsNullOrEmpty($TexFile)) {
    # No arguments - compile all papers
    Compile-AllPapers
} else {
    # Specific file provided
    if (Test-Path $TexFile) {
        Compile-Paper $TexFile
    } else {
        Write-Host "❌ Error: File $TexFile not found" -ForegroundColor Red
        exit 1
    }
}

Write-Host "LaTeX compilation complete!" -ForegroundColor Green
