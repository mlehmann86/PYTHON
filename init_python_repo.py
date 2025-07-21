import os
import requests
import subprocess

# === CONFIGURATION ===
github_username = "mlehmann"  # change if different
repo_name = "PYTHON"
description = "Personal Python analysis scripts and tools"
private = True  # change to False if you want it public
token = os.environ.get("GITHUB_TOKEN")  # token stored securely in env var
local_path = "/theory/lts/mlehmann/PYTHON"

# === GitHub Repo Creation ===
url = "https://api.github.com/user/repos"
headers = {"Authorization": f"token {token}"}
payload = {
    "name": repo_name,
    "description": description,
    "private": private
}

print(f"📡 Creating GitHub repo '{repo_name}'...")
response = requests.post(url, json=payload, headers=headers)
if response.status_code == 201:
    print("✅ GitHub repo created.")
else:
    print(f"❌ Failed to create GitHub repo: {response.json().get('message')}")
    exit(1)

# === Initialize Local Git Repo ===
print(f"🛠️  Initializing local repo at {local_path}...")
os.chdir(local_path)
subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "remote", "add", "origin", f"git@github.com:{github_username}/{repo_name}.git"], check=True)

# Create a standard .gitignore (customize later if needed)
with open(".gitignore", "w") as f:
    f.write("__pycache__/\n*.pyc\n*.npz\n*.pdf\n.DS_Store\n")

subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
subprocess.run(["git", "branch", "-M", "master"], check=True)
subprocess.run(["git", "push", "-u", "origin", "master"], check=True)

print("🚀 Sync complete. Local repo pushed to GitHub.")
