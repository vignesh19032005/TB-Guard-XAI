import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "mistral-hackaton-2026/TB-Guard-XAI"
print(f"🚀 Deploying to HuggingFace Spaces: {REPO_ID}")

# Prompt for user details
hf_token = input("\n🔑 Please paste your Hugging Face Write Token: ").strip()

# Initialize API
api = HfApi(token=hf_token)

try:
    # Create Space (if it doesn't exist)
    print(f"📦 Creating/Verifying Space: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print("✅ Space created/verified")
    
    # Upload files using upload_folder (much easier for whole projects)
    print("📤 Uploading files... (This might take a few minutes because of the model file)")
    
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".vscode/*",
            "__pycache__/*",
            "datasets_raw/*",
            "datasets_processed/*",
            "data/*",
            "venv/*",
            ".env",
            "*.log",
            "deploy_to_hf.py",  # no need to upload this script
            "archive/*",  # exclude validation datasets
            "validation_results/*",  # exclude validation results
            "batch_temp/*",  # exclude batch temp files
            "batch_reports/*",  # exclude batch reports
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "pip-log.txt",
            "pip-delete-this-directory.txt",
            ".pytest_cache/*",
            "*.egg-info/*",
            "dist/*",
            "build/*",
            ".DS_Store",
            "Thumbs.db",
            "analyze_validation_failure.py",  # exclude analysis scripts
            "validate_dadb_dataset.py",
            "validate_with_threshold_tuning.py",
            "VALIDATION_SUMMARY.md",
            "EXTERNAL_VALIDATION.md"
        ]
    )
    
    print("\n✅ Deployment complete!")
    print(f"🌐 Your API will be available at: https://huggingface.co/spaces/{REPO_ID}")
    print("\n⏳ Note: First deployment may take 5-10 minutes to build and start")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("If you are getting an Unauthorized error, please run this in your terminal:")
    print("    huggingface-cli login")
    print("And paste a token with 'Write' permissions for the mistral-hackaton-2026 organization.")
