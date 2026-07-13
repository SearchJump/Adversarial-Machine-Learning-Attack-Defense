import os
import urllib.request

def setup_assets():
    print("==================================================")
    print("          AI RED TEAMING PORTFOLIO ASSET SETUP     ")
    print("==================================================")
    
    # 1. Create root directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("my_assessment", exist_ok=True)
    os.makedirs("src", exist_ok=True)
    
    # 2. Download ImageNet labels to data/labels.txt
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_dest = "data/labels.txt"
    if not os.path.exists(labels_dest):
        print(f"Downloading ImageNet labels to {labels_dest}...")
        try:
            urllib.request.urlretrieve(labels_url, labels_dest)
            print("Labels downloaded successfully.")
        except Exception as e:
            print(f"Failed to download labels: {e}")
    else:
        print("Labels file already exists.")

    # 3. Download a sample rooster image for Evasion.py
    rooster_url = "https://upload.wikimedia.org/wikipedia/commons/5/52/Brown_Leghorn_rooster_in_Australia.jpg"
    rooster_dest = "src/test_image.png"
    if not os.path.exists(rooster_dest):
        print(f"Downloading sample rooster image to {rooster_dest}...")
        try:
            urllib.request.urlretrieve(rooster_url, rooster_dest)
            print("Rooster image downloaded successfully.")
        except Exception as e:
            print(f"Failed to download rooster image: {e}")
    else:
        print("Rooster image already exists.")

    # 4. Create Symbolic Links inside src/ to enable transparent execution
    # This prevents path failures by mapping src/data -> data and src/my_assessment -> my_assessment
    symlinks = [
        ("src/data", "../data"),
        ("src/my_assessment", "../my_assessment")
    ]
    for src_link, target in symlinks:
        if not os.path.exists(src_link):
            try:
                os.symlink(target, src_link)
                print(f"Created symlink: {src_link} -> {target}")
            except Exception as e:
                print(f"Could not create symlink {src_link}: {e}")
        else:
            print(f"Link or directory {src_link} already exists.")
            
    print("\nSetup complete. You are now ready to run the validation pipeline!")
    print("==================================================\n")

if __name__ == "__main__":
    setup_assets()
