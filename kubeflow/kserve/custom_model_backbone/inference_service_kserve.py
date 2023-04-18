import kserve
from typing import Dict
import logging

class Model(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        # self.load()

    def load(self):
        pass

    def predict(self, request: Dict) -> Dict:
        logging.info("Payload: %s", request)

        return {"prediction": [1, 2, 3]}


if __name__ == "__main__":
    model = Model("custom-model")
    kserve.KFServer().start([model])
