# 📰 Fake News Detection Using NLP 🚫🖥️

![fake-news.jpg](https://img.freepik.com/free-photo/newspaper-background-concept_23-2149501641.jpg)

This project leverages **Natural Language Processing (NLP)** techniques to detect fake news in articles and online content. With the rise of misinformation on digital platforms, identifying and filtering out fake news has become a crucial task. Our project aims to automatically classify news articles as *real* or *fake* by analyzing the text content, providing a scalable and efficient tool for addressing this global challenge.


## Project Structure 🗂️

Below is the detailed explanation of the files and directories included in this project:

### Root Directory

- **embedding\_generation.ipynb**: 📄 Jupyter Notebook for generating text embeddings from the provided dataset. This notebook preprocesses the data and creates embeddings that are essential for training the model.

- **model\_training.ipynb**: 📊 Jupyter Notebook that handles the training of the machine learning model using the generated embeddings. It includes data loading, training, validation, and evaluation. Additionally, MLflow is integrated for experiment tracking and model versioning. 🔍📈

- **docker-compose.yml**: ⚙️ Configuration file for Docker Compose that sets up the required services and dependencies for the project.

- **poetry.lock**: 🔒 Lock file generated by Poetry to ensure consistency of installed dependencies.

- **pyproject.toml**: 📝 The project configuration file used by Poetry, specifying project dependencies and metadata.

- **.gitignore**: 🚫 Specifies files and directories that should be ignored by Git, such as temporary files, data caches, and environment configurations.

### `app/` Directory 📁

- **main.py**: 🖥️ The main Python script that sets up the API endpoint. This API handles requests to the model for predicting whether the news is fake or legitimate.

### `data/` Directory 📂

- **features/**: 🗃️ Contains the feature data used in the model training and evaluation.
- **labels/**: 🏷️ Contains the corresponding labels for the feature data.

## Installation and Setup 🛠️

To get started with this project, follow these steps:

### Prerequisites 📋

- Python 3.8+ 🐍
- Docker 🐳
- Poetry (for dependency management) 📦
- MLflow (for experiment tracking) 📊

### Installation Steps 🔧

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd fake-news-detector
   ```

2. **Install dependencies using Poetry**:

   ```bash
   poetry install
   ```

3. **Set up Docker environment**:

   ```bash
   docker-compose up --build
   ```

### Running Jupyter Notebooks 🚀

To explore or execute the notebooks:

1. **Activate Poetry shell**:
   ```bash
   poetry shell
   ```
2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## MLflow Integration 🔗

The project includes MLflow for experiment tracking and model versioning. This helps in:

- **Tracking Experiments**: Keep track of different model runs and their parameters, metrics, and outputs.
- **Model Versioning**: Manage and deploy different versions of the trained model seamlessly.

To use MLflow:

1. **Run MLflow Server**:
   ```bash
   mlflow server --host 0.0.0.0 --backend-store-uri ./mlruns --artifacts-destination ./mlartifacts --dev
   ```
   Access the UI at `http://localhost:5000` to monitor and compare experiment results.

2. **Log Experiments**: Model training in `model_training.ipynb` is configured to log metrics and parameters to MLflow automatically. 📈

## API Usage 🌐

The API for predicting fake news is defined in `app/main.py`. To run the API:

1. Ensure that the environment is up and running with Streamlit:
   ```bash
   streamlit run app/main.py
   ```
2. Access the API at `http://127.0.0.1:5001/invocations` for model predictions.

### Example Request 📝

You can use `call_model_api.ipynb` to test the API. This notebook demonstrates how to make API calls and handle responses.

## Project Workflow 🔄

1. **Data Preparation**: 🧹 Ensure that data is correctly placed in `data/features/` and `data/labels/`.
2. **Embedding Generation**: 🛠️ Run `embedding_generation.ipynb` to create text embeddings.
3. **Model Training**: 🏋️ Execute `model_training.ipynb` for training and validating the machine learning model.
4. **Experiment Tracking**: 🔍 Log and monitor experiments with MLflow.
5. **API Deployment**: 🌍 Deploy and interact with the model via the API as described above.

## Contributing 🤝

Feel free to open issues or submit pull requests for improvements or bug fixes. Please ensure that your changes are well-documented 📝 and tested ✅.


## Contact 📧

For questions or suggestions, please reach out to Antonio at antonioj.oliver.garcia@gmail.com

---

Happy Coding! 💻✨🍀


