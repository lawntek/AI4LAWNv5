# import os
# import subprocess
# import urllib.request
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# os.makedirs('./models', exist_ok=True)
# # 1. Install Detectree2 (only once)
# # def install_detectree2():
# #     try:
# #         import detectree2
# #     except ImportError:
# #         subprocess.check_call(["pip", "install", "git+https://github.com/patball1/detectree2.git"])

# # 2. Download model weights if not already downloaded
# def download_model_weights():
#     model_path = "./models/250312_flexi.pth"
#     if not os.path.exists(model_path):
#         print("Downloading model weights...")
#         url = "https://zenodo.org/records/15014353/files/250312_flexi.pth?download=1"
#         urllib.request.urlretrieve(url, model_path)
#         print("Download complete.")
#     else:
#         print("Model already downloaded.")

# # 3. Call setup functions
# # install_detectree2()
# download_model_weights()
