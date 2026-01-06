# Trading System Documentation

Welcome to the Trading System documentation. This directory contains comprehensive documentation for users, developers, and contributors.

## 📚 Documentation Structure

### For Users

- **[Getting Started Guide](user_guide/getting_started.md)** - Installation, first backtest, and basic usage
- **[User Guide Index](user_guide/README.md)** - Complete user documentation index
- **[Example Configurations](../EXAMPLE_CONFIGS/README.md)** - Example configuration files with explanations
- **[n8n Integration Guide](N8N_INTEGRATION.md)** - Workflow automation with n8n

### For Developers

- **[API Reference](api/index.rst)** - Complete API documentation (Sphinx-generated)
- **[Architecture Documentation](../agent-files/01_ARCHITECTURE_OVERVIEW.md)** - System design and architecture
- **[Developer Guide](developer_guide/README.md)** - Development setup, testing, and contribution guidelines

### Technical Documentation

- **[Architecture & Design](../agent-files/)** - Comprehensive technical documentation
  - [Architecture Overview](../agent-files/01_ARCHITECTURE_OVERVIEW.md)
  - [Configuration Guide](../agent-files/02_CONFIGS_AND_PARAMETERS.md)
  - [Data Pipeline](../agent-files/03_DATA_PIPELINE_AND_VALIDATION.md)
  - [Indicators Library](../agent-files/04_INDICATORS_LIBRARY.md)
  - [Portfolio & Risk](../agent-files/05_PORTFOLIO_AND_RISK.md)
  - [Strategy Details](../agent-files/06_STRATEGY_EQUITY.md) (Equity)
  - [Strategy Details](../agent-files/07_STRATEGY_CRYPTO.md) (Crypto)
  - [Backtest Engine](../agent-files/10_BACKTEST_ENGINE.md)
  - [Validation Suite](../agent-files/12_VALIDATION_SUITE.md)
  - [CLI Commands](../agent-files/15_CLI_COMMANDS.md)
  - [And more...](../agent-files/)

### Testing & Validation

- **[Testing Guide](../TESTING_GUIDE.md)** - Comprehensive testing instructions
- **[Quick Start Testing](../QUICK_START_TESTING.md)** - Quick testing reference
- **[Test Fixtures](../tests/fixtures/README.md)** - Test data documentation

### Project Documentation

- **[Main README](../README.md)** - Project overview and quick start
- **[Review Summary](../agent-files/REVIEW_SUMMARY.md)** - Codebase review and status
- **[Next Steps](../agent-files/NEXT_STEPS.md)** - Roadmap and future enhancements

## 🗺️ Quick Navigation

### I want to...

- **Get started quickly**: [Getting Started Guide](user_guide/getting_started.md)
- **Understand the architecture**: [Architecture Overview](../agent-files/01_ARCHITECTURE_OVERVIEW.md)
- **Configure a strategy**: [Configuration Guide](../agent-files/02_CONFIGS_AND_PARAMETERS.md)
- **Run tests**: [Testing Guide](../TESTING_GUIDE.md)
- **See API reference**: [API Documentation](api/index.rst)
- **Find examples**: [Example Configurations](../EXAMPLE_CONFIGS/README.md)
- **Understand strategies**: [Equity Strategy](../agent-files/06_STRATEGY_EQUITY.md) | [Crypto Strategy](../agent-files/07_STRATEGY_CRYPTO.md)
- **Learn about validation**: [Validation Suite](../agent-files/12_VALIDATION_SUITE.md)
- **Automate with n8n**: [n8n Integration Guide](N8N_INTEGRATION.md)

## 📖 Documentation Locations

Documentation is organized across three main locations:

1. **`docs/`** (this directory)
   - User guides
   - Developer guides
   - API reference (Sphinx)
   - Organized documentation structure

2. **`agent-files/`**
   - Architecture and design documentation
   - Technical specifications
   - Implementation details
   - Algorithm pseudocode
   - Agent task specifications
   - Implementation status documents

3. **Root directory**
   - Main README
   - Testing guides
   - Deployment documentation
   - User-facing references

## 🔍 Building API Documentation

To build the Sphinx API documentation:

```bash
cd docs/
make html
```

The generated documentation will be in `docs/_build/html/`.

## 📝 Contributing to Documentation

When adding or updating documentation:

1. **User-facing guides**: Add to `docs/user_guide/`
2. **Developer guides**: Add to `docs/developer_guide/`
3. **API documentation**: Update docstrings in code, then regenerate Sphinx docs
4. **Architecture docs**: Add to `agent-files/` if technical, or `docs/` if user-facing

## 🔗 External Links

- [Project Repository](../README.md)
- [Example Configurations](../EXAMPLE_CONFIGS/)
- [Test Suite](../tests/)

---

**Last Updated**: 2024-12-19
