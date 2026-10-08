# sites_cube

## Install git

### Linux

```bash
sudo apt install git
```

### Windows

```powershell
winget install Git.Git
```

## Clone

### SSH

```bash
git clone git@github.com:kwundram2602/sites_cube.git
cd sites_cube
```

### HTTPS

```bash
git clone https://github.com/kwundram2602/sites_cube.git
cd sites_cube
```

## Update

```bash
git pull
```

## Install uv

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Sync

```bash
uv sync
```

## Run

```bash
uv run sites-cube config=config/sentinel2.yaml
```
