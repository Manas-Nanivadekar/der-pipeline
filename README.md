# DER calc pipeline

```bash
uv sync
uv install
cp .env.example .env
# Update .env with your settings
export HF_TOKEN="you_hf_token_here"
python3 main.py # make sure data.csv is present in the same directory
```
