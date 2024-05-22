import pandas as pd

def preprocess_data(data_path: str):
    # Load the data
    data = pd.read_excel(data_path)

    # Drop rows with NaN values
    data = data.dropna()

    # Separate questions and traits
    # Convert questions into a single text column
    questions = data[['Q1', 'Q2', 'Q3']].fillna('').agg(' '.join, axis=1)
    traits = data[['work', 'teachability', 'commitment', 'flexibility', 'adventerous']]
    
    return questions, traits


