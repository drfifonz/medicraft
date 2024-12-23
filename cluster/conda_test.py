import sys

import torch

RUN_ID = int(sys.argv[1])

print("Starting task: ", RUN_ID)

models = {
    # "reference": ".results/0003/reference/model-50.pt",
    "benign": "medicraft-models/benign/model-75.pt",
    "fluid": "medicraft-models/fluid/model-75.pt",
    "precancerous": "medicraft-models/precancerous/model-75.pt",
}


JOB_LIST = {
    1: "benign",
    2: "fluid",
    3: "fluid",
    4: "precancerous",
    5: "precancerous",
}

START_SAMPLE_IDX = 0 if RUN_ID in [1, 2, 4] else 1001


current_model_name = JOB_LIST[RUN_ID]
model, model_path = current_model_name, models[current_model_name]
print(RUN_ID, current_model_name, model_path)
print("RANGE: ", START_SAMPLE_IDX, "-", START_SAMPLE_IDX + 1000)

print("\nCuda avalible:", torch.cuda.is_available())
