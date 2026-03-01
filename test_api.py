import requests
from pathlib import Path

file_path = Path(r"datasets_raw/kaggle_tb/TB_Chest_Radiography_Database/Normal/Normal-19.png")
with open(file_path, "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data={"symptoms": "", "threshold": 0.500}
    )

print("Status:", response.status_code)
if response.status_code == 200:
    res = response.json()
    print("Prediction:", res["prediction"])
    print("Probability:", res["probability"])
    print("Uncertainty:", res["uncertainty"])
    print("Std:", res["uncertainty_std"])
    print("Region:", res.get("gradcam_region", res.get("region")))
    print("Explanation:", res.get("clinical_explanation"))
    print("Evidence length:", len(res.get("evidence", [])))
else:
    print(response.text)
