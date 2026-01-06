# Trading System Documentation Index

This is the main documentation index for the Trading System. All documentation is organized and linked from here.

## 📚 Documentation Locations

Documentation is organized across three main locations:

1. **`docs/`** - Organized user and developer documentation
2. **`agent-files/`** - Architecture, technical design, agent tasks, and implementation status
3. **Root directory** - Project overview, testing guides, deployment docs, and user-facing references

## 🗺️ Quick Navigation

### For Users

- **[Getting Started](docs/user_guide/getting_started.md)** - Installation and first backtest
- **[User Guide Index](docs/user_guide/README.md)** - Complete user documentation
- **[Example Configurations](EXAMPLE_CONFIGS/README.md)** - Working configuration examples
- **[n8n Integration](docs/N8N_INTEGRATION.md)** - Workflow automation with n8n
- **[Main README](README.md)** - Project overview

### For Developers

- **[Developer Guide](docs/developer_guide/README.md)** - Development setup and guidelines
- **[API Reference](docs/api/index.rst)** - Complete API documentation (Sphinx)
- **[Architecture Overview](agent-files/01_ARCHITECTURE_OVERVIEW.md)** - System design
- **[Code Style Guide](docs/developer_guide/code_style.md)** - Coding standards

### Technical Documentation

- **[Architecture & Design](agent-files/)** - Comprehensive technical documentation
  - [Architecture Overview](agent-files/01_ARCHITECTURE_OVERVIEW.md)
  - [Configuration Guide](agent-files/02_CONFIGS_AND_PARAMETERS.md)
  - [Data Pipeline](agent-files/03_DATA_PIPELINE_AND_VALIDATION.md)
  - [Indicators Library](agent-files/04_INDICATORS_LIBRARY.md)
  - [Portfolio & Risk](agent-files/05_PORTFOLIO_AND_RISK.md)
  - [Equity Strategy](agent-files/06_STRATEGY_EQUITY.md)
  - [Crypto Strategy](agent-files/07_STRATEGY_CRYPTO.md)
  - [Backtest Engine](agent-files/10_BACKTEST_ENGINE.md)
  - [Validation Suite](agent-files/12_VALIDATION_SUITE.md)
  - [CLI Commands](agent-files/15_CLI_COMMANDS.md)
  - [And more...](agent-files/)

### Testing & Validation

- **[Testing Guide](TESTING_GUIDE.md)** - Comprehensive testing instructions
- **[Quick Start Testing](QUICK_START_TESTING.md)** - Quick testing reference
- **[Test Fixtures](tests/fixtures/README.md)** - Test data documentation

### Troubleshooting & Reference

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Error Code Reference](ERROR_CODE_REFERENCE.md)** - Comprehensive error code reference guide
- **[FAQ](FAQ.md)** - Frequently asked questions

### Project Status

- **[Review Summary](agent-files/REVIEW_SUMMARY.md)** - Codebase review and status
- **[Production Readiness](PRODUCTION_READINESS.md)** - Production deployment checklist
- **[Next Steps](agent-files/NEXT_STEPS.md)** - Roadmap and future enhancements

## 📖 Documentation Structure

```
trade-test/
├── docs/                          # Organized documentation
│   ├── README.md                 # Documentation index
│   ├── user_guide/               # User-facing guides
│   │   ├── README.md
│   │   └── getting_started.md
│   ├── developer_guide/          # Developer documentation
│   │   ├── README.md
│   │   ├── development_setup.md
│   │   └── code_style.md
│   ├── api/                      # API reference (Sphinx)
│   └── index.rst                 # Sphinx main index
│
├── agent-files/                  # Architecture, design, & agent docs
│   ├── 01_ARCHITECTURE_OVERVIEW.md
│   ├── 02_CONFIGS_AND_PARAMETERS.md
│   ├── 03_DATA_PIPELINE_AND_VALIDATION.md
│   ├── AGENT_TASKS_PHASE*.md     # Agent task specifications
│   ├── REVIEW_SUMMARY.md         # Codebase review
│   ├── COMPLETE_SYSTEM_VISION.md  # System vision document
│   └── ... (30+ files)
│
├── README.md                      # Main project README
├── TESTING_GUIDE.md              # Testing documentation
├── QUICK_START_TESTING.md        # Quick testing reference
└── DOCUMENTATION.md              # This file
```

## 🔍 Finding Documentation

### I want to...

- **Get started quickly**: [Getting Started Guide](docs/user_guide/getting_started.md)
- **Understand the architecture**: [Architecture Overview](agent-files/01_ARCHITECTURE_OVERVIEW.md)
- **Configure a strategy**: [Configuration Guide](agent-files/02_CONFIGS_AND_PARAMETERS.md)
- **Run tests**: [Testing Guide](TESTING_GUIDE.md)
- **See API reference**: [API Documentation](docs/api/index.rst)
- **Find examples**: [Example Configurations](EXAMPLE_CONFIGS/README.md)
- **Understand strategies**: [Equity Strategy](agent-files/06_STRATEGY_EQUITY.md) | [Crypto Strategy](agent-files/07_STRATEGY_CRYPTO.md)
- **Learn about validation**: [Validation Suite](agent-files/12_VALIDATION_SUITE.md)
- **Automate with n8n**: [n8n Integration Guide](docs/N8N_INTEGRATION.md)
- **Troubleshoot errors**: [Troubleshooting Guide](TROUBLESHOOTING.md) | [Error Code Reference](ERROR_CODE_REFERENCE.md)
- **Set up development**: [Developer Guide](docs/developer_guide/README.md)
- **Check project status**: [Review Summary](agent-files/REVIEW_SUMMARY.md) | [Production Readiness](PRODUCTION_READINESS.md)

## 📝 Building Documentation

### Sphinx API Documentation

To build the Sphinx API documentation:

```bash
cd docs/
make html
```

The generated documentation will be in `docs/_build/html/`.

**Note**: Requires `myst_parser` for Markdown support:
```bash
pip install myst-parser
```

## 🔗 External Resources

- [Project Repository](README.md)
- [Example Configurations](EXAMPLE_CONFIGS/)
- [Test Suite](tests/)

---

**Last Updated**: 2024-12-19
