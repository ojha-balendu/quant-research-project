import os
import paramiko
import shutil
from datetime import datetime

def archive_targets(local_path):
    """Saves a timestamped copy of the submitted CSV for record keeping."""
    archive_dir = os.path.join(os.path.dirname(local_path), "submitted_archives")
    os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"targets_{timestamp}.csv"
    archive_path = os.path.join(archive_dir, filename)
    
    shutil.copy2(local_path, archive_path)
    print(f"ARCHIVE: Saved local copy to {archive_path}")

def upload_targets_file(local_path, region, username, private_key_path, host="sftp.qrt.cloud"):
    # 1. Archive the file locally before uploading
    archive_targets(local_path)
    
    # 2. Upload to SFTP
    try:
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load private key: {e}")
        return

    transport = paramiko.Transport((host, 22))
    
    try:
        print(f"Connecting to {host}:22")
        transport.connect(username=username, pkey=private_key)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        filename = os.path.basename(local_path)
        remote_path = f"incoming/{region.lower()}/{filename}"
        
        print(f"Uploading {filename} to {remote_path}...")
        sftp.put(local_path, remote_path)
        print(f"SUCCESS: File successfully uploaded to {region} server.")
        
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"CRITICAL ERROR: SFTP Upload failed: {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGETS_DIR = os.path.join(BASE_DIR, "targets")
    target_file = os.path.join(TARGETS_DIR, "targets.csv")
    
    if not os.path.exists(target_file):
        print("ERROR: targets.csv not found!")
        exit(1)
        
    USERNAME = os.environ.get("SFTP_USERNAME")
    KEY_PATH = "private_key.pem"
    
    upload_targets_file(target_file, "AMER", USERNAME, KEY_PATH)
