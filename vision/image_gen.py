import subprocess
import os

class ImageGen:
    def __init__(self, config):
        self.output_dir = "./data/output/"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_from_text(self, text):
        # Example command for stable-diffusion.cpp (edit paths as needed)
        cmd = [
            "./stable-diffusion", "--prompt", text,
            "--output", os.path.join(self.output_dir, "generated.png")
        ]
        subprocess.run(cmd)
