from src.data_generator import generate_student_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Starting Student Performance Prediction System...")

    generate_student_data()
    preprocess_data()
    train_model()
    evaluate_model()

    print("Project executed successfully.")

if __name__ == "__main__":
    main()