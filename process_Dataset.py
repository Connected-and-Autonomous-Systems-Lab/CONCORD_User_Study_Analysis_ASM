

from pathlib import Path
import tempfile
import rarfile

from edge_detect import process_maze_folder  # your existing folder-based function



# 👇 adjust this path to where your UnRAR.exe actually is
rarfile.UNRAR_TOOL = r".\UnRAR.exe"
# make sure it uses the extract command interface
rarfile.USE_EXTRACT_CMD = True


# Change this if your dataset root is elsewhere
DATASET_ROOT = Path("../Collaboration_user_study")
OUTPUT_ROOT = Path("./output")


def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset root not found: {DATASET_ROOT.resolve()}")
        return

    # Iterate over each user folder inside dataset
    for user_dir in sorted(DATASET_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        username = user_dir.name
        rar_path = user_dir / f"{username}.rar"

        if not rar_path.exists():
            print(f"[WARN] No RAR file found for user '{username}' at {rar_path}")
            continue

        # For each difficulty level we care about
        for level in ("Easy", "Hard"):
            print(f"\n=== User: {username} | Level: {level} ===")

            # 1) Make a temporary directory that will be auto-deleted
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # 2) Extract the whole RAR into the temp dir
                try:
                    with rarfile.RarFile(rar_path) as rf:
                        rf.extractall(tmpdir)
                except rarfile.Error as e:
                    print(f"[ERROR] Failed to extract {rar_path}: {e}")
                    continue

                # 3) Build path to Unity folder inside the extracted tree
                unity_folder = tmpdir / username / level / "Unity"

                if not unity_folder.exists():
                    print(f"[WARN] Unity folder not found at {unity_folder}")
                    continue

                # 4) Build output folder for this user + level
                out_dir = OUTPUT_ROOT / username / level

                # 5) Call your existing process_maze_folder on the real folder
                process_maze_folder(
                    str(unity_folder),   # input_folder (images)
                    str(out_dir),        # output_folder
                    debug=False          # or True if you want debug images
                )
            # 6) When we exit the 'with tempfile.TemporaryDirectory()' block,
            #    the entire temp folder is deleted automatically.


if __name__ == "__main__":
    main()
