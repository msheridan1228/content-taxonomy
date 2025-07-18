from abc import ABC, abstractmethod
## Placeholder for now
### Want to instantiate a base class that can be extended for different tag naming methods
# Keep methods consistent using this base class
class TagNamer:
    def __init__(self, docs_df):
        self.docs_df = docs_df

    def check_required_columns(self, required_columns):
        for column in required_columns:
            if column not in self.docs_df.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def fit_predict(self):
        self.fit()
        return self.predict()