#!/bin/bash

# Code formatting script for the RAG chatbot project
# This script runs Black formatter and isort for consistent code formatting

echo "🔧 Running code formatting..."

echo "📝 Formatting Python files with Black..."
uv run black backend/ main.py

echo "📋 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"