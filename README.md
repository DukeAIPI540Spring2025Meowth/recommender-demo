# recommender-demo
Recommender System demo

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the setup script to load and transform data:
```bash
python setup.py
```

5. To deactivate the virtual environment when you're done:
```bash
deactivate
```

## Running the Evaluation

To evaluate the recommender system, run:
```bash
python -m scripts.eval.evaluate
```

This will run the evaluation module and output the performance metrics of the recommender system.
