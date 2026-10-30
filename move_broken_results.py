import shutil
import datetime
from pathlib import Path


def backup_recent_outputs():
    # 1. Configuration
    # src_dir = Path("vggt_outputs")
    # dst_dir = Path("vggt_outputs_backup")
    # target_filename = "stat.json"

    # src_dir = Path("../gsplat/results/vggt_outputs")
    # dst_dir = Path("../gsplat/results/vggt_outputs_backup")
    # target_filename = "stats/val_step6999.json"

    src_dir = Path("../gsplat/results/colmap_outputs")
    dst_dir = Path("../gsplat/results/colmap_outputs_backup")
    target_filename = "stats/val_step6999.json"

    # Set the threshold date (Year, Month, Day, Hour, Minute)
    # 16 April 2026 at 14:59
    threshold_date = datetime.datetime(2026, 4, 15, 9, 00, 0)
    threshold_timestamp = threshold_date.timestamp()

    # 2. Pre-flight checks
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    # Create the backup directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 3. Iterate through folders and move if conditions are met
    moved_count = 0
    for folder in src_dir.iterdir():
        if folder.is_dir():
            target_file = folder / target_filename

            # Check if the specific file exists in the folder
            if target_file.exists():
                # Get the file's timestamp
                # .st_mtime is used for cross-platform reliability (last written/modified)
                file_timestamp = target_file.stat().st_mtime

                # Check if the file was created/modified after the threshold
                if file_timestamp > threshold_timestamp:
                    file_time_str = datetime.datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Moving '{folder.name}'... (stat.json timestamp: {file_time_str})")

                    destination_path = dst_dir / folder.name

                    try:
                        shutil.move(str(folder), str(destination_path))
                        moved_count += 1
                    except Exception as e:
                        print(f"Failed to move '{folder.name}': {e}")

    print(f"\nOperation complete. Moved {moved_count} folder(s).")


if __name__ == "__main__":
    backup_recent_outputs()
