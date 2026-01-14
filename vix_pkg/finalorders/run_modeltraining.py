import subprocess

def main():
    # Call your model training and prediction script
    subprocess.run(["python", "vix_pkg/copy_train_six_cmf_model.py"], check=True)

if __name__ == "__main__":
    main()