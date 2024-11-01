# CynthiaSubnet

A Bittensor subnet implementation for Cynthia Systems that revolutionizes AI-powered search and discovery through decentralized cognitive search, delivering unparalleled performance, trust, and scalability.

![Cynthia Subnet](docs/CynthiaSubnet.jpg)

## Overview

CynthiaSubnet leverages Bittensor's decentralized network to create a powerful search and discovery infrastructure. The subnet validates and rewards nodes based on search relevance, response latency, and result consistency.

### Features

- Semantic search processing using transformer models
- Advanced relevance scoring system
- Performance-based incentive mechanism
- Trust and safety validation
- Scalable architecture

## Installation

CynthiaSubnet uses Poetry for dependency management. To get started:

```bash
# Create a virtual environment (Optional)
python3 -m venv ~/ai
source ~/ai/bin/activate

# Install Poetry if you haven't already
pip install poetry

# Clone the repository
git clone git@github.com:cynthiasystems/CynthiaSubnet.git
cd CynthiaSubnet

# Install dependencies
poetry install
```

## Quick Start

1. Create and activate a new Python virtual environment
2. Install dependencies using Poetry
3. Configure your Bittensor wallet
4. Run validator or miner node

### Running a Validator

```bash
poetry run python neurons/validator.py --wallet.name <wallet_name> --wallet.hotkey <hotkey_name>
```

### Running a Miner

```bash
poetry run python neurons/miner.py --wallet.name <wallet_name> --wallet.hotkey <hotkey_name>
```

## Development

To contribute to CynthiaSubnet:

1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

- Website: https://cynthiasystems.com
- Email: info@cynthiasystems.com
