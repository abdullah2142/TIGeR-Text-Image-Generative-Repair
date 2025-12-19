import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # tiny pandas check
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    print("Pandas OK:\n", df)

    # load a CLIP model (just to ensure transformers works)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Transformers CLIP loaded OK:", model_name)

if __name__ == "__main__":
    main()
