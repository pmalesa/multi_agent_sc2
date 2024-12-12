# Multi Agent SC2
The aim of the project is to train a team of agents playing StarCraft II, on different scenarios, using the DQN algorithm with VDN and QMIX decompositions.

## Virtual Environment Activation
```bash
# Initialize the environment
conda init

# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate multi_agent_sc2_env

# Reload the environment
conda env update --name multi_agent_sc2_env --file environment.yml --prune
```

## StarCraft 2 Installation (Linux)
```bash
# Run the installation script
./install_sc2.sh
```

## Virtual Environment Removal
```bash
# Deactivate the environment
conda deactivate

# Remove the existing environment
conda remove --name multi_agent_sc2_env --all
```