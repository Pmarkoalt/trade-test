Trading System Documentation
============================

Welcome to the Trading System documentation. This documentation provides a comprehensive reference for all modules, classes, and functions in the trading system.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   README
   user_guide/getting_started
   developer_guide/README
   api/index

Overview
--------

The Trading System is a config-driven daily momentum trading system for equities and cryptocurrency with walk-forward backtesting, realistic execution costs, and comprehensive validation suite.

Key Components
--------------

* **Backtest Engine**: Walk-forward backtesting with event-driven daily loop
* **Strategies**: Equity and crypto momentum strategies
* **Data Pipeline**: Data loading, validation, and feature computation
* **Portfolio Management**: Position sizing, risk controls, and portfolio tracking
* **Execution**: Realistic execution simulation with slippage and fees
* **Validation**: Statistical tests and stress scenarios
* **Reporting**: Performance metrics and trade logging

Documentation Structure
-----------------------

* :doc:`README <README>` - Documentation index and navigation
* :doc:`user_guide/getting_started` - Getting started guide for users
* :doc:`developer_guide/README` - Developer guide index
* :doc:`api/index` - Complete API reference

Additional Resources
--------------------

* `Main README <../README.md>`_ - Project overview
* `Architecture Documentation <../agent-files/>`_ - Technical specifications
* `Example Configurations <../EXAMPLE_CONFIGS/>`_ - Configuration examples
* `Testing Guide <../TESTING_GUIDE.md>`_ - Testing instructions

API Reference
-------------

The complete API reference is organized by module:

.. toctree::
   :maxdepth: 2

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

