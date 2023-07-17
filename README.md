# blangchain


## Setup
1. Run the following
```
conda env create -f env.yml
conda develop .
```

2. Make sure your OPENAI_API_KEY environment variable is set (e.g. in your bash profile).
```
export OPENAI_API_KEY="..."
```
You can get a key from your account page: https://platform.openai.com/account/api-keys


## Example usage
```
python example_scripts/compositional_entailment.py
```
