import pickle
import numpy as np

# Cargar el modelo entrenado
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Función para predecir el riesgo de infarto
def predict_heart_attack(age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin):
    # Convertir las entradas a un array de NumPy
    features = np.array([[age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin]])
    
    # Hacer la predicción
    prediction = model.predict(features)
    
    # Interpretar la predicción
    if prediction[0] == 1:
        return "Positive (Risk of Heart Attack)"
    else:
        return "Negative (No Risk of Heart Attack)"

# Ejemplo de uso
if __name__ == "__main__":
    # Solicitar los inputs del usuario
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (1 for male, 0 for female): "))
    impluse = int(input("Enter pulse: "))
    pressurehight = int(input("Enter high blood pressure: "))
    pressurelow = int(input("Enter low blood pressure: "))
    glucose = float(input("Enter glucose level: "))
    kcm = float(input("Enter KCM: "))
    troponin = float(input("Enter troponin level: "))

    # Realizar la predicción
    result = predict_heart_attack(age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin)
    print(f"The prediction for the given inputs is: {result}")
