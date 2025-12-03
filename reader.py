

import rarfile


def readFileNames(rar_path: str, unity_folder_inside_rar: str) -> None:
    
    unity_folder_inside_rar = unity_folder_inside_rar.rstrip("/") + "/"

    with rarfile.RarFile(rar_path) as rf:
        print(f"\nFiles in Unity folder: {unity_folder_inside_rar} (RAR: {rar_path})")

        # Loop through all entries inside the RAR
        for info in rf.infolist():
            # Skip directories
            if info.is_dir():
                continue

            # Check if this file is inside the Unity folder
            if info.filename.startswith(unity_folder_inside_rar):
                # Get only the filename (last part)
                filename_only = info.filename.split("/")[-1]
                print(filename_only)
