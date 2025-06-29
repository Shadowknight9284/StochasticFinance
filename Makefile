# Makefile for Quantitative Strategy Research System
# ================================================

# Default target
.PHONY: all papers cpp clean help

# Variables
STRATEGY_DIRS := $(wildcard strategies/*)
PAPER_PDFS := $(addsuffix /paper.pdf, $(STRATEGY_DIRS))
CPP_EXECUTABLES := $(foreach dir,$(STRATEGY_DIRS),$(dir)/$(notdir $(dir))_strategy)

# Default target - build everything
all: papers cpp

# Build all LaTeX papers
papers: $(PAPER_PDFS)

# Build all C++ executables
cpp: $(CPP_EXECUTABLES)

# Pattern rule for building PDFs from LaTeX
strategies/%/paper.pdf: strategies/%/paper.tex style.tex
	@echo "Building LaTeX paper: $<"
	@cd $(dir $<) && \
	pdflatex -interaction=nonstopmode $(notdir $<) > /dev/null && \
	pdflatex -interaction=nonstopmode $(notdir $<) > /dev/null && \
	pdflatex -interaction=nonstopmode $(notdir $<) > /dev/null
	@cd $(dir $<) && rm -f *.aux *.log *.out *.toc *.bbl *.blg *.run.xml *.bcf *.fdb_latexmk *.fls *.synctex.gz
	@echo "✅ Created: $@"

# Pattern rule for building C++ executables
strategies/%/%_strategy: strategies/%/*.cpp strategies/%/*.hpp
	@echo "Building C++ strategy: $@"
	@cd $(dir $@) && \
	if [ -d build ]; then rm -rf build; fi && \
	mkdir -p build && \
	cd build && \
	cmake .. -DCMAKE_BUILD_TYPE=Release && \
	make -j$(shell nproc 2>/dev/null || echo 4)
	@echo "✅ Built: $@"

# Individual strategy targets
ornstein-uhlenbeck: strategies/OrnsteinUhlenbeck/paper.pdf strategies/OrnsteinUhlenbeck/ornsteinuhlenbeck_strategy

heston: strategies/HestonVolSurface/paper.pdf strategies/HestonVolSurface/hestonvolsurface_strategy

jump-diffusion: strategies/JumpDiffusion/paper.pdf strategies/JumpDiffusion/jumpdiffusion_strategy

log-normal: strategies/LogNormalJumpMeanReversion/paper.pdf strategies/LogNormalJumpMeanReversion/lognormaljumpmeanreversion_strategy

# Run demonstrations
demo:
	@echo "🚀 Running system demonstration..."
	python demo.py

# Validate all strategies
validate:
	@echo "🔍 Validating all strategies..."
	@for dir in $(STRATEGY_DIRS); do \
		if [ -f "$$dir/paper.pdf" ] && [ -f "$$dir/build/$$strategy_strategy" ]; then \
			echo "✅ $$dir: Complete"; \
		else \
			echo "❌ $$dir: Missing components"; \
		fi; \
	done

# Clean all build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@for dir in $(STRATEGY_DIRS); do \
		cd "$$dir" && rm -rf build/ *.pdf *.aux *.log *.out *.toc *.bbl *.blg *.run.xml *.bcf *.fdb_latexmk *.fls *.synctex.gz; \
		cd ..; \
	done
	@rm -rf __pycache__/ *.pyc
	@echo "✅ Clean complete"

# Clean only LaTeX auxiliary files, keep PDFs
clean-latex:
	@echo "🧹 Cleaning LaTeX auxiliary files..."
	@find strategies -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
	      -name "*.bbl" -o -name "*.blg" -o -name "*.run.xml" -o -name "*.bcf" -o \
	      -name "*.fdb_latexmk" -o -name "*.fls" -o -name "*.synctex.gz" | xargs rm -f
	@echo "✅ LaTeX cleanup complete"

# Statistics about the repository
stats:
	@echo "📊 Repository Statistics:"
	@echo "Strategies: $(words $(STRATEGY_DIRS))"
	@echo "LaTeX Papers: $(shell find strategies -name "paper.pdf" | wc -l)/$(words $(STRATEGY_DIRS))"
	@echo "C++ Headers: $(shell find strategies -name "*.hpp" | wc -l)"
	@echo "C++ Sources: $(shell find strategies -name "*.cpp" | wc -l)"
	@echo "Python Files: $(shell find . -maxdepth 1 -name "*.py" | wc -l)"
	@echo "Total Lines of Code:"
	@echo "  LaTeX: $(shell find strategies -name "*.tex" -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "  C++: $(shell find strategies -name "*.cpp" -o -name "*.hpp" -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "  Python: $(shell find . -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $$1}')"

# Install dependencies
install-deps:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Python dependencies installed"

# Setup development environment
setup: install-deps
	@echo "🔧 Setting up development environment..."
	@if [ ! -d ".git" ]; then git init; fi
	@echo "✅ Development environment ready"

# Create new strategy
new-strategy:
	@read -p "Strategy name: " name; \
	read -p "SDE framework: " sde; \
	read -p "Asset universe: " universe; \
	python -c "from quant_research_system import QuantResearchSystem; \
	           system = QuantResearchSystem('.'); \
	           result = system.create_strategy('$$name', '$$sde', '$$universe'); \
	           print('✅ Strategy created:', result['strategy_name'])"

# Help target
help:
	@echo "🎯 Quantitative Strategy Research System - Make Targets"
	@echo "======================================================="
	@echo ""
	@echo "📄 LaTeX Targets:"
	@echo "  papers              Build all LaTeX papers"
	@echo "  clean-latex         Clean LaTeX auxiliary files"
	@echo ""
	@echo "💻 C++ Targets:"
	@echo "  cpp                 Build all C++ executables"
	@echo ""
	@echo "🏗️  Build Targets:"
	@echo "  all                 Build everything (papers + cpp)"
	@echo "  clean               Clean all build artifacts"
	@echo ""
	@echo "🎯 Strategy Targets:"
	@echo "  ornstein-uhlenbeck  Build Ornstein-Uhlenbeck strategy"
	@echo "  heston              Build Heston strategy"
	@echo "  jump-diffusion      Build Jump Diffusion strategy"
	@echo "  log-normal          Build Log-Normal strategy"
	@echo ""
	@echo "🚀 Development Targets:"
	@echo "  demo                Run system demonstration"
	@echo "  validate            Validate all strategies"
	@echo "  new-strategy        Create new strategy interactively"
	@echo "  setup               Setup development environment"
	@echo "  stats               Show repository statistics"
	@echo ""
	@echo "📦 Utility Targets:"
	@echo "  install-deps        Install Python dependencies"
	@echo "  help                Show this help message"

# Force rebuild targets
.PHONY: demo validate clean clean-latex stats install-deps setup new-strategy help
