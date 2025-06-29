#!/bin/bash
# LaTeX Build Script for Strategy Papers
# =====================================
# This script compiles LaTeX papers and cleans up auxiliary files,
# keeping only the final PDF output.

# Function to compile a single LaTeX paper
compile_paper() {
    local tex_file=$1
    local paper_dir=$(dirname "$tex_file")
    local paper_name=$(basename "$tex_file" .tex)
    
    echo "🔨 Compiling: $tex_file"
    echo "📁 Directory: $paper_dir"
    echo "📄 Output: $paper_name.pdf"
    
    # Change to paper directory
    cd "$paper_dir"
    
    # First compilation pass
    echo "📝 First pass..."
    pdflatex -interaction=nonstopmode "$paper_name.tex" > /dev/null 2>&1
    
    # Check if bibliography exists and compile if needed
    if [ -f "$paper_name.bbl" ] || grep -q "\\bibliography" "$paper_name.tex"; then
        echo "📚 Processing bibliography..."
        bibtex "$paper_name" > /dev/null 2>&1
        pdflatex -interaction=nonstopmode "$paper_name.tex" > /dev/null 2>&1
    fi
    
    # Final compilation pass
    echo "📝 Final pass..."
    pdflatex -interaction=nonstopmode "$paper_name.tex" > /dev/null 2>&1
    
    # Check if PDF was generated successfully
    if [ -f "$paper_name.pdf" ]; then
        echo "✅ Success: $paper_name.pdf generated"
        
        # Clean up auxiliary files
        echo "🧹 Cleaning auxiliary files..."
        rm -f *.aux *.log *.out *.toc *.bbl *.blg *.run.xml *.bcf \
              *.fdb_latexmk *.fls *.synctex.gz *.nav *.snm *.vrb \
              *.figlist *.makefile *.dvi *.ps *.lof *.lot *.acn \
              *.acr *.alg *.glg *.glo *.gls *.ist *.loa *.xdy
        
        echo "🎉 Build complete: $(pwd)/$paper_name.pdf"
    else
        echo "❌ Error: Failed to generate PDF"
        echo "📋 Check LaTeX errors in log files"
        return 1
    fi
    
    # Return to original directory
    cd - > /dev/null
}

# Function to compile all strategy papers
compile_all_papers() {
    echo "🚀 Compiling all strategy papers..."
    
    # Find all paper.tex files in strategies subdirectories
    for tex_file in strategies/*/paper.tex; do
        if [ -f "$tex_file" ]; then
            compile_paper "$tex_file"
            echo ""
        fi
    done
    
    echo "📊 Summary of generated PDFs:"
    find strategies -name "*.pdf" -type f | while read pdf; do
        echo "  📄 $pdf"
    done
}

# Main execution
if [ $# -eq 0 ]; then
    # No arguments - compile all papers
    compile_all_papers
elif [ $# -eq 1 ]; then
    # Single argument - compile specific paper
    if [ -f "$1" ]; then
        compile_paper "$1"
    else
        echo "❌ Error: File $1 not found"
        exit 1
    fi
else
    echo "Usage: $0 [paper.tex]"
    echo "  No arguments: Compile all strategy papers"
    echo "  With argument: Compile specific paper"
    exit 1
fi
