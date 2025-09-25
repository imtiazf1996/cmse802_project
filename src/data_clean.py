import pandas as pd

def clean_data(input_file, output_file):
    """
    Placeholder cleaning function.
    Reads CSV, prints shape, and saves copy.
    """
    df = pd.read_csv(input_file)
    print("Initial shape:", df.shape)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    print("Data cleaning step (to be implemented)")
